# Instructions on Benchmark
This is a very messy package I used for experimentation and quick insights. I need to clean this.
The benchmark figure in the paper is obtained from:       
```
best_bench_time.py
perf_vis.py
```

## Memory profiling:
1. Make sure that the run dir is empty:      
```
rm -r /tmp/tensorboard_*
```
2. Run the memory test script to generate the memory trace:    
```
XLA_PYTHON_CLIENT_MEM_FRACTION=.99 python mem_test.py
```
3.  Start tensorboard webserver and make a local connection to remote if necessary
```
tensorboard --logdir /tmp/tensorboard_20000/ --port 9000
ssh -N -f -L localhost:9003:localhost:9001 ubuntu@130.162.165.233 -i /home/rares/.ssh/gpu_1_oracle.priv
```
4. Go to Memory viewer -> select the function you want to view and select the latest run (biggest timestamp). To be able to view at the function level in jax, all the functions need to be jit-ed. 