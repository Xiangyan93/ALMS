#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from ..args import MonitorArgs


class BaseTask(ABC):
    @abstractmethod
    def active_learning(self, args: MonitorArgs):
        pass

    @abstractmethod
    def create(self, args: MonitorArgs):
        pass

    @abstractmethod
    def build(self, args: MonitorArgs):
        pass

    @abstractmethod
    def run(self, args: MonitorArgs):
        pass

    @abstractmethod
    def analyze(self, args: MonitorArgs):
        pass

    @abstractmethod
    def extend(self, args: MonitorArgs):
        pass

    @abstractmethod
    def update_fail_tasks(self):
        pass
