import numpy as np
from abc import ABC, abstractmethod
from math import exp
import QuantLib as ql
from pyspark import SparkContext
from pyspark.mllib.random import RandomRDDs

sc = SparkContext(appName='CVA')


class AbstractCva(ABC):
    def __init__(self, nb_instruments, nb_scenarios, nb_timesteps):
        self.nb_instruments = nb_instruments
        self.nb_scenarios = nb_scenarios
        self.nb_timesteps = nb_timesteps
        self._timesteps = None

    @abstractmethod
    def build_scenarios(self):
        pass

    @abstractmethod
    def age_and_price(self, randArrayRDD):
        pass

    @abstractmethod
    def aggregate(self, cube_rdd, curve_today):
        '''
        cube: np array of array, axis=0 : scenario, axis=1 : timestep
        timesteps: list of timestep
        returns: array of NI values
        '''
        pass

    def run(self, curve_today):
        scenarios = self.build_scenarios()
        pv_cube = self.age_and_price(scenarios)
        cva = self.aggregate(pv_cube, curve_today)
        print('********************** ' + str(cva))


class DummyCva(AbstractCva):

    @property
    def timesteps(self):
        if not self._timesteps:
            self._timesteps = list(range(self.nb_timesteps))
        return self._timesteps

    def build_pv_cube(self):
        pass

    def age_and_price(self, randArrayRDD):
        '''
        gaussians: is an ndarrray of gaussian
        nb_instruments: number of instruments to swap
        returns a 2D-array: scenario by instruments
        '''
        nb_instruments = self.nb_instruments

        def fast_dummy_price(p):
            return np.ones(nb_instruments) * np.transpose([p])

        def dummy_price(p):
            res = fast_dummy_price(p)
            dummy_factor = np.arange(1, nb_instruments+1) * np.identity(nb_instruments)
            return np.dot(res, dummy_factor)

        cube = randArrayRDD.map(fast_dummy_price)
        return cube

    def build_scenarios(self):
        return RandomRDDs.normalVectorRDD(sc, self.nb_scenarios, self.nb_timesteps, seed=1)


class DummyAggregateCVAMixin(object):
    def aggregate(self, cube_rdd, curve_today):
        ee = cube_rdd.reduce(lambda a, b: a + b)  # NT*NI
        ee = np.array(ee) * 1.0 / self.nb_scenarios
        cva = np.sum(ee, axis=0)
        return cva


class AggregateCVAMixin(object):
    def aggregate(self, cube_rdd):
        S = 0.05
        R = 0.4
        def positive(p):
            p[p < 0] = 0
            return p
        cube_rdd = cube_rdd.map(positive)
        ee = cube_rdd.reduce(lambda a, b: a + b)  # NT*NI
        ee = np.array(ee) * 1.0 / self.nb_scenarios
        cva = 0
        T = self.timesteps
        for i in range(len(T) - 1):
            temp = exp(-S * T[i] / (1.0 - R)) - exp(-S * T[i + 1] / (1.0 - R))
            temp *= ee[i] + ee[i + 1]
            # cva += 0.5 * crvToday.discount(T[i+1]) * temp
            cva += 0.5 * temp  # TODO add the discount factor
        cva *= 1.0 - R
        return cva


class BetterDummyCva(DummyCva):
    def aggregate(self, cube_rdd, curve_today):
        S = 0.05
        R = 0.4

        def positive(p):
            p[p < 0] = 0
            return p
        cube_rdd = cube_rdd.map(positive)
        ee = cube_rdd.reduce(lambda a, b: a + b)  # NT*NI
        ee = np.array(ee) * 1.0 / self.nb_scenarios
        cva = 0
        T = self.timesteps
        for i in range(len(T) - 1):
            temp = exp(-S * T[i] / (1.0 - R)) - exp(-S * T[i + 1] / (1.0 - R))
            temp *= ee[i] + ee[i + 1]
            cva += 0.5 * curve_today.discount(T[i+1]) * temp
        cva *= 1.0 - R
        return cva

todaysDate=ql.Date(26,12,2013)
ql.Settings.instance().evaluationDate=todaysDate
crvTodaydates=[ql.Date(26,12,2013),
               ql.Date(30,6,2014),
               ql.Date(30,7,2014),
               ql.Date(29,8,2014),
               ql.Date(30,9,2014),
               ql.Date(30,10,2014),
               ql.Date(28,11,2014),
               ql.Date(30,12,2014),
               ql.Date(30,1,2015),
               ql.Date(27,2,2015),
               ql.Date(30,3,2015),
               ql.Date(30,4,2015),
               ql.Date(29,5,2015),
               ql.Date(30,6,2015),
               ql.Date(30,12,2015),
               ql.Date(30,12,2016),
               ql.Date(29,12,2017),
               ql.Date(31,12,2018),
               ql.Date(30,12,2019),
               ql.Date(30,12,2020),
               ql.Date(30,12,2021),
               ql.Date(30,12,2022),
               ql.Date(29,12,2023),
               ql.Date(30,12,2024),
               ql.Date(30,12,2025),
               ql.Date(29,12,2028),
               ql.Date(30,12,2033),
               ql.Date(30,12,2038),
               ql.Date(30,12,2043),
               ql.Date(30,12,2048),
               ql.Date(30,12,2053),
               ql.Date(30,12,2058),
               ql.Date(31,12,2063)]
crvTodaydf=[1.0,
            0.998022,
            0.99771,
            0.99739,
            0.997017,
            0.996671,
            0.996337,
            0.995921,
            0.995522,
            0.995157,
            0.994706,
            0.994248,
            0.993805,
            0.993285,
            0.989614,
            0.978541,
            0.961973,
            0.940868,
            0.916831,
            0.890805,
            0.863413,
            0.834987,
            0.807111,
            0.778332,
            0.750525,
            0.674707,
            0.575192,
            0.501258,
            0.44131,
            0.384733,
            0.340425,
            0.294694,
            0.260792
            ]

curve_today = ql.DiscountCurve(crvTodaydates, crvTodaydf, ql.Actual360(), ql.TARGET())
#DummyCva(NI, NS, NT).run()
# randArrayRDD.saveAsTextFile("hdfs://dbhtlx009.dns21.socgen/user/automation/matencio")

config_40GB = {'nb_instruments' : 200000, 'nb_scenarios' : 1000, 'nb_timesteps' : 50}
config_1GB = {'nb_instruments' : 5000, 'nb_scenarios' : 1000, 'nb_timesteps' : 50}
config_20MB = {'nb_instruments' : 1000, 'nb_scenarios' : 100, 'nb_timesteps' : 50}

BetterDummyCva(**config_40GB).run(curve_today)
