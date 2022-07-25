import os
import time

gpu_resource = [10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 100]
batch = [1, 8, 16, 32]

model_name = ["alexnet"]
for m in model_name:
    for g in gpu_resource:
        os.system("echo quit | sudo nvidia-cuda-mps-control")
        time.sleep(1)
        os.system("sudo nvidia-cuda-mps-control -d")
        time.sleep(1)
        os.system("echo set_default_active_thread_percentage {} | sudo nvidia-cuda-mps-control".format(g))
        time.sleep(1)
        for b in batch:
            print("-----------------------gpu = {}, batch = {} -------------------------".format(g, b))
            resfile_name = "{}_b{}_loop1_g{}".format(m, b, g)
            python_file_name = "{}_b{}.py".format(m, b)
            cmd = "sudo /usr/local/NVIDIA-Nsight-Compute-2022.2/target/linux-desktop-glibc_2_11_3-x64/ncu --export /home/hpj/hpj/Triton/5.25test/kernel/auto_profile/{}.nsight-cuprof-report.ncu-rep --force-overwrite --target-processes application-only --replay-mode kernel --kernel-name-base function --launch-skip-before-match 0 --section ComputeWorkloadAnalysis --section InstructionStats --section LaunchStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Tables --section Nvlink_Tables --section Nvlink_Topology --section Occupancy --section SchedulerStats --section SourceCounters --section SpeedOfLight --section SpeedOfLight_RooflineChart --section WarpStateStats --sampling-interval auto --sampling-max-passes 5 --sampling-buffer-size 33554432 --profile-from-start 1 --cache-control all --clock-control base --apply-rules yes --import-source no --check-exit-code yes /home/hpj/anaconda3/bin/python /home/hpj/hpj/Triton/5.25test/kernel/auto_profile/{}".format(resfile_name, python_file_name)
            os.system(cmd)

            time.sleep(3)
        


