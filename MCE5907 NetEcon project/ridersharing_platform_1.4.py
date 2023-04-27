# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :   ridersharing_platform_1.4.py
@Time    :   2023/04/25 17:13:58
@Author  :   Liao Hongjie
@Version :   1.4
"""

# 增加了total revenue的计算

import numpy as np
import matplotlib.pyplot as plt

SCORE_PARAM_BASEPRICE = 0.8
SCORE_PARAM_BONUS = 1


class RidesharePlatform:
    """
    代表共享出行平台，包括平台的基本属性（如司机数量、起步价、最大等待时间、乘客到达率等）,
    以及平台的运行方法（step方法和run方法）。
    """

    def __init__(self, num_drivers, max_wait_time, lambda_value, charge_ratio, base_price=1):
        self.num_drivers = num_drivers  # 司机数量
        self.base_price = base_price  # 起步价
        self.max_wait_time = max_wait_time  # 乘客能接受的最大等待时间
        self.lambda_value = lambda_value  # 乘客到达率
        self.charge_ratio = charge_ratio  # 按距离收费倍率
        self.drivers = [Driver(np.random.uniform(0, 1))
                        for _ in range(num_drivers)]  # 司机列表
        self.riders = []  # 乘客列表
        self.history = []  # 历史订单列表

    def step(self):
        for driver in self.drivers:
            driver.update()

        for rider in self.riders:
            best_driver = None
            best_score = float('-inf')
            for driver in self.drivers:
                if driver.status == 'available':
                    score = rider.score(driver, self.charge_ratio)
                    if score > best_score:
                        best_driver = driver
                        best_score = score
            if best_driver is not None:
                ride = Ride(rider, best_driver)
                self.history.append(ride)
                self.riders.remove(rider)
                best_driver.status = 'unavailable'
                best_driver.destination = rider.destination

        # 按照possion分布生成乘客
        if np.random.poisson(self.lambda_value) > 0:
            origin = np.random.uniform(0, 10)
            destination = np.random.uniform(0, 10)
            rider = Rider(origin, destination)
            self.riders.append(rider)

    def run(self, num_steps):
        for i in range(num_steps):
            self.step()

    def plot_order_curve(self, num_points=40):
        prices = np.linspace(
            self.base_price, 2 * self.base_price, num_points)
        order = []
        for price in prices:
            count = 0
            for rider in self.riders:
                if rider.score(None, self.charge_ratio) + SCORE_PARAM_BASEPRICE*self.base_price >= price:
                    count += 1
            order.append(count)
        plt.plot(prices, order)
        plt.xlabel('Price')
        plt.ylabel('Order')
        plt.title(
            f'Order Curve with charge_ratio={self.charge_ratio}')
        plt.show()

    def plot_revenue_curve(self, num_points=40):
        prices = np.linspace(
            self.base_price, 2 * self.base_price, num_points)
        order = []
        revenues = []
        for price in prices:
            count = 0
            for rider in self.riders:
                if rider.score(None, self.charge_ratio) + SCORE_PARAM_BASEPRICE*self.base_price >= price:
                    count += 1
            revenue = price * count
            order.append(count)
            revenues.append(revenue)
        plt.plot(prices, revenues)
        plt.xlabel('Price')
        plt.ylabel('Revenue')
        plt.title(
            f'Revenue Curve with charge_ratio={self.charge_ratio}')
        plt.show()

    def plot_order_curve_with_bonus(self, num_points=40, times_of_max_bonus=0.7, num_of_bonus=8):
        bonus_range = (0, times_of_max_bonus * self.base_price)
        prices = np.linspace(self.base_price, 2 * self.base_price, num_points)
        bonuses = np.linspace(bonus_range[0], bonus_range[1], num_of_bonus)
        order = []
        for bonus in bonuses:
            order_bonus = []
            for price in prices:
                count = 0
                for rider in self.riders:
                    if rider.score(None, self.charge_ratio) + SCORE_PARAM_BONUS*bonus + SCORE_PARAM_BASEPRICE*self.base_price >= price:
                        count += 1
                order_bonus.append(count)
            order.append(order_bonus)
        order = np.array(order)

        fig, ax = plt.subplots()
        for i in range(num_of_bonus):
            ax.plot(prices, order[i], label=f"Bonus={bonuses[i]:.2f}")
        ax.legend(loc="best")
        ax.set_xlabel('Price')
        ax.set_ylabel('Order')
        ax.set_title(
            f'Order Curve with Bonus and charge_ratio={self.charge_ratio}')
        plt.show()

    def plot_revenue_curve_with_bonus(self, num_points=40, times_of_max_bonus=0.7, num_of_bonus=8):
        bonus_range = (0, times_of_max_bonus * self.base_price)
        prices = np.linspace(self.base_price, 2 * self.base_price, num_points)
        bonuses = np.linspace(bonus_range[0], bonus_range[1], num_of_bonus)
        order = []
        for bonus in bonuses:
            order_bonus = []
            for price in prices:
                count = 0
                for rider in self.riders:
                    if rider.score(None, self.charge_ratio) + SCORE_PARAM_BONUS*bonus + SCORE_PARAM_BASEPRICE*self.base_price >= price:
                        count += 1
                order_bonus.append(count)
            order.append(order_bonus)
        order = np.array(order)

        fig, ax = plt.subplots()
        for i in range(num_of_bonus):
            revenue = (prices + bonuses[i]) * order[i]
            ax.plot(prices, revenue, label=f"Bonus={bonuses[i]:.2f}")
        ax.legend(loc="best")
        ax.set_xlabel('Price')
        ax.set_ylabel('Revenue')
        ax.set_title(
            f'Revenue Curve with Bonus and charge_ratio={self.charge_ratio}')
        plt.show()


class Driver:
    """
    代表司机，包括司机的位置、状态（可用或不可用）和目的地等属性，以及根据目的地更新位置的方法。
    """

    def __init__(self, position):
        self.position = position
        self.status = 'available'
        self.destination = None
        self.dispatch = False
        self.score = 0

    def update(self):
        if self.status == 'unavailable':
            distance_to_destination = abs(self.destination - self.position)
            if distance_to_destination < 0.01:
                self.status = 'available'
                self.destination = None
            else:
                if self.destination > self.position:
                    self.position += 0.01
                else:
                    self.position -= 0.01


class Rider:
    """
    代表乘客，包括起始点和目的地等属性，以及计算乘客得分（根据距离愿意支付的意愿）的方法。
    """

    def __init__(self, origin, destination, fee_per_distance=1):
        self.origin = origin
        self.destination = destination
        self.fee_per_distance = fee_per_distance

    def score(self, driver, charge_ratio):
        if driver is None:
            distance = abs(self.destination - self.origin)
        else:
            distance = abs(driver.position - self.origin) + \
                abs(self.destination - self.origin)
        score_fee = self.fee_per_distance * charge_ratio * distance
        return score_fee


class Ride:
    """
    代表一次乘车记录，包括乘客和司机
    """

    def __init__(self, rider, driver):
        self.rider = rider
        self.driver = driver


if __name__ == '__main__':

    # 标准费率
    platform_1 = RidesharePlatform(
        num_drivers=20, base_price=10, max_wait_time=10, lambda_value=20, charge_ratio=1)
    platform_1.run(1000)
    # 高峰期更高费率=1.3
    platform_2 = RidesharePlatform(
        num_drivers=20, base_price=10, max_wait_time=10, lambda_value=20, charge_ratio=1.3)
    platform_2.run(1000)

    platform_1.plot_order_curve(num_points=40)
    platform_1.plot_revenue_curve(num_points=40)
    platform_2.plot_order_curve(num_points=40)
    platform_2.plot_revenue_curve(num_points=40)

    platform_1.plot_order_curve_with_bonus(num_points=40)
    platform_1.plot_revenue_curve_with_bonus(num_points=40)

    platform_2.plot_order_curve_with_bonus(num_points=40)
    platform_2.plot_revenue_curve_with_bonus(num_points=40)
