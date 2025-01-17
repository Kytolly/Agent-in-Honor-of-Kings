#!/usr/bin/env python3
# -*- coding:utf-8 -*-

'''
@Project :1v1
@File    :sgame_rl_helper.py
@Author  :kaiwu
@Date    :2022/10/20 11:43 

'''

import traceback
from framework.server.aisrv.kaiwu_rl_helper import KaiWuRLHelper
from framework.interface.exception import SkipEpisodeException, ClientQuitException, TimeoutEpisodeException
from framework.common.config.config_control import CONFIG
from framework.common.utils.common_func import TimeIt
from framework.common.utils.kaiwudrl_define import KaiwuDRLDefine

class SgameRLHelper(KaiWuRLHelper):

    def run(self) -> None:
        try:
            self.env.init()
        except AssertionError as e:
            self.logger.error(f"kaiwu_rl_helper self.env.init() {str(e)}, traceback.print_exc() is {traceback.format_exc()}")
            self.env.reject(e)
        except Exception as e:
            self.logger.error(f"kaiwu_rl_helper self.env.init() {str(e)}, traceback.print_exc() is {traceback.format_exc()}")
            self.env.reject(e)
        else:
            self.client_id = self.env.client_id
            try:
                self.agent_ctxs = {}
                self.simu_ctx.agent_ctxs = self.agent_ctxs

                def run_episode_once():
                    self.run_episode()

                while not self.exit_flag.value:
                    try:
                        run_episode_once()

                    except SkipEpisodeException:
                        self.logger.error("kaiwu_rl_helper run_episode_once() SkipEpisodeException {}", str(e))
                        pass
            
            except ClientQuitException:
                self.logger.error(f"kaiwu_rl_helper run_episode_once() ClientQuitException {str(e)}, traceback.print_exc() is {traceback.format_exc()}")
            except TimeoutEpisodeException as e:
                self.logger.error(f"kaiwu_rl_helper run_episode_once() TimeoutEpisodeException {str(e)}, traceback.print_exc() is {traceback.format_exc()}")
                self.env.reject(e)
            except AssertionError as e:
                self.logger.error(f"kaiwu_rl_helper run_episode_once() AssertionError {str(e)}, traceback.print_exc() is {traceback.format_exc()}")
                self.env.reject(e)
            except Exception as e:
                self.logger.error(f"kaiwu_rl_helper run_episode_once() Exception {str(e)}, traceback.print_exc() is {traceback.format_exc()}")
                if not self.exit_flag.value:
                    self.env.reject(e)
        finally:
            self.logger.info("kaiwu_rl_helper finally")
            self.stop()
         

    def run_episode(self):

        with TimeIt() as ti:
            self.sgame_1v1_episode_main_loop()

    def sgame_1v1_episode_main_loop(self):

        while not self.exit_flag.value:
            try:
                states, must_need_sample_info = self.env.next_valid()
                if states:
                    valid_agents = list(states.keys())

                    # 判断游戏结束, 生成和发送样本
                    if self.env.run_handler.done:
                        self.logger.info('kaiwu_rl_helper game is over')
                        # 样本生成器是单例模式，里面包含所有Agent生成的样本，因此只需要调用一次
                        agent_id = valid_agents[0]
                        agent_ctx = self.agent_ctxs[agent_id]
                        for policy_id in agent_ctx.policy:
                            if agent_ctx.policy[policy_id].need_train():
                                self.reward_value = self.gen_train_data(agent_id, policy_id)
                                # 传入上报数据
                                self.data_queue.append(self.reward_value)
                            # self.logger.debug("kaiwu_rl_helper gen_train_data success")
                            if agent_ctx.done:
                                self.stop_agent(agent_id)
                        # 游戏结束后，不需要预测
                        break

                    # 准备数据
                    with TimeIt() as ti:
                        for agent_id in valid_agents:
                            if agent_id not in self.agent_ctxs:
                                self.start_agent(agent_id)

                            agent_ctx = self.agent_ctxs[agent_id]
                            agent_ctx.state, agent_ctx.pred_input = {}, {}

                            policy_id = agent_ctx.main_id
                            s = states[agent_id].get_state()
                            agent_ctx.pred_input[policy_id] = s
                            agent_ctx.state[policy_id] = states[agent_id]

                    # self.logger.debug("kaiwu_rl_helper prepare Msg success")

                    # 执行预测
                    self.predict(valid_agents)
                    # self.logger.debug("kaiwu_rl_helper predict success")

                    # 处理action, 保留样本
                    format_action_list = []
                    network_sample_info_list = []
                    lstm_cell_list, lstm_hidden_list =[],[]
                    for agent_id in valid_agents:
                        agent_ctx = self.agent_ctxs[agent_id]
                        for policy_id in agent_ctx.policy:
                            format_action = agent_ctx.pred_output[policy_id][agent_id]['format_action']
                            network_sample_info = agent_ctx.pred_output[policy_id][agent_id]['network_sample_info']
                            lstm_info = agent_ctx.pred_output[policy_id][agent_id]['lstm_info']
                            format_action_list.append(format_action)
                            network_sample_info_list.append(network_sample_info[0])
                            lstm_cell_list.append(lstm_info[0][0])
                            lstm_hidden_list.append(lstm_info[0][1])

                    self.env.on_handle_action(format_action_list)

                    #  在train模式下才需要保留样本
                    if CONFIG.run_mode == KaiwuDRLDefine.RUN_MODEL_TRAIN:
                        # 每次action 操作后, 保留样本
                        self.gen_expr(valid_agents[0], self.agent_ctxs[valid_agents[0]].main_id, {
                            'network_sample_info': network_sample_info_list,
                            'must_need_sample_info': must_need_sample_info
                        })
                        self.env.run_handler.update_lstm(lstm_cell_list,lstm_hidden_list)
                        # self.logger.debug("kaiwu_rl_helper handle action and gen_expr success")

                        if self.steps>0 and self.steps%int(CONFIG.send_sample_size)==0:
                            self.logger.info('kaiwu_rl_helper send samples to learner during gaming')
                            agent_id = valid_agents[0]
                            agent_ctx = self.agent_ctxs[agent_id]
                            for policy_id in agent_ctx.policy:
                                if agent_ctx.policy[policy_id].need_train():
                                    self.reward_value = self.gen_train_data(agent_id, policy_id)
                                    # 传入上报数据
                                    self.data_queue.append(self.reward_value)
                    
                    self.steps += 1
                
                else:
                    if must_need_sample_info == "end":
                        #主动退出循环
                        self.exit_flag.value = True
                        
                        #  在train模式下才需要保留样本
                        if CONFIG.run_mode == KaiwuDRLDefine.RUN_MODEL_TRAIN:
                            # 处理异常退出情况,保存样本
                            if not self.env.run_handler.done:
                                # 样本生成器是单例模式，里面包含所有Agent生成的样本，因此只需要调用一次
                                if self.agent_ctxs:
                                    agent_id = list(self.agent_ctxs.keys())[0]
                                    agent_ctx = self.agent_ctxs[agent_id]
                                    for policy_id in agent_ctx.policy:
                                        if agent_ctx.policy[policy_id].need_train():
                                            self.reward_value = self.gen_train_data(agent_id, policy_id, del_last=True)
                                    self.logger.info("kaiwu_rl_helper gen_train_data success")

                        ##结束aisrv的循环
                        self.env.msg_buff.output_q.put(None)

            except AssertionError as e:
                self.logger.error(f"kaiwu_rl_helper sgame_1v1_episode_main_loop {str(e)}, traceback.print_exc() is {traceback.format_exc()}")
                self.env.reject(e)
                break
            except ClientQuitException as e:
                self.logger.error(f"kaiwu_rl_helper sgame_1v1_episode_main_loop ClientQuitException {str(e)}, traceback.print_exc() is {traceback.format_exc()}")
                break
            except TimeoutEpisodeException as e:
                self.logger.error(f"kaiwu_rl_helper sgame_1v1_episode_main_loop TimeoutEpisodeException {str(e)}, traceback.print_exc() is {traceback.format_exc()}")
                self.env.reject(e)
                break
            except Exception as e:
                self.logger.error(f"kaiwu_rl_helper sgame_1v1_episode_main_loop {str(e)}, traceback.print_exc() is {traceback.format_exc()}")
                self.env.reject(e)
                break