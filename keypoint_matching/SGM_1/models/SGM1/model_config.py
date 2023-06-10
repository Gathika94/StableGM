from easydict import EasyDict as edict

__C = edict()

model_cfg = __C

# SGM1 model options
__C.SGM1 = edict()
__C.SGM1.SOLVER_NAME = 'LPMP'
__C.SGM1.LAMBDA_VAL = 80.0
__C.SGM1.SOLVER_PARAMS = edict()
__C.SGM1.SOLVER_PARAMS.timeout = 1000
__C.SGM1.SOLVER_PARAMS.primalComputationInterval = 10
__C.SGM1.SOLVER_PARAMS.maxIter = 100
__C.SGM1.FEATURE_CHANNEL = 1024
__C.SGM1.SK_ITER_NUM = 20
__C.SGM1.SK_EPSILON = 1.0e-10
__C.SGM1.SK_TAU = 0.003