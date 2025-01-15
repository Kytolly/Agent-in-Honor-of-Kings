#!/usr/bin/env python3
# -*- coding:utf-8 -*-

'''
@Project :1v1
@File    :learner.py
@Author  :kaiwu
@Date    :2022/6/15 20:57 

'''

from framework.common.utils.tf_utils import *
from framework.server.learner.on_policy_trainer import OnPolicyTrainer
from framework.common.config.config_control import CONFIG

'''
PPO Trainer
'''
class Sgame1V1PPOTrainer(OnPolicyTrainer):

    def __init__(self):
        super(Sgame1V1PPOTrainer, self).__init__(name='ppo')
    
    def init(self):
        super().init()

    @property
    def tensor_names(self):
        '''
        目前sgame的设置一个属性input_datas
        设置reverb表中数据的key
        发送样本时, 每条样本为一个dict, key为tensor_names
        '''
        names = []
        names.append('input_datas')
        return names

    @property
    def tensor_dtypes(self):
        """设置样本的类型"""
        dtypes = []
        dtypes.append(tf.float16)
        return dtypes

    @property
    def tensor_shapes(self):
        """设置样本的shape"""
        shapes = []
        shapes.append(tf.TensorShape((64, 15552)))

        return shapes