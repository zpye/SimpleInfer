import simpleinfer as infer
import numpy as np

infer.InitializeContext()

engine = infer.Engine()

rc = engine.LoadModel('res.param', 'res.bin')
