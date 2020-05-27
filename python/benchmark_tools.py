import datetime
import subprocess

from benchmark_paths import root


def run(defines, prefix, profiling_path):
    def exec_cmd(cmd, directory=""):
        print(cmd)
        subprocess.check_call(cmd, cwd=root + "/" + directory, shell=True)

    def check_define(define):
        exec_cmd("                                   "
                 "if [[ $(grep {} ./src -r -w | wc -c) -eq 0 ]]; "
                 "then echo; echo Invalid define: {}; echo; exit 1; fi".format(define, define))

    prefix = datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S') + "_" + prefix

    defines.append(("STATS_FILES_PREFIX", '\"{}\"'.format(prefix)))
    defines.append(("PROFILING_PATH", '\"{}\"'.format(profiling_path)))
    defines.append(("BENCHMARK", '1'))
    defines.append(("EXIT_AFTER_REPLAY", '1'))

    with open(profiling_path + "/" + prefix + ".defines", "w") as file:
        for name, value in defines:
            line = "{}={}\n".format(name, value)
            file.write(line)

    with open(root + "/src/script_definitions.h", "w") as file:
        for name, value in defines:
            check_define(name)
            line = "#define {} {}\n".format(name, value)
            file.write(line)
            print(line, end="")

    exec_cmd("cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc ../", "cmake-build-release")
    exec_cmd("cmake --build ./cmake-build-release --target DAG_edits_demo -- -j 8")
    exec_cmd("./cmake-build-release/DAG_edits_demo")

    with open(root + "/src/script_definitions.h", "w") as file:
        pass
