import os

def get_setting_str(key, value):
    setting_str = " --" + str(key) + " " + str(value)
    return setting_str

def run_diff_history_len(dataset, mode):
    len_list = [6]
    for l in len_list:
        command = "python main.py"
        command += get_setting_str("device", "cuda:1")
        command += get_setting_str("dataset", dataset)
        command += get_setting_str("description", dataset+"-history-len-"+str(l)+"-"+mode+",")
        command += get_setting_str("history-len", l)
        command += get_setting_str("mode", mode)
        if dataset == "ICEWS14":
            command += get_setting_str("use-valid", "false")
            command += get_setting_str("max-epochs", 15)
        print(command)
        os.system(command)
        
if __name__ == '__main__':
    # run_diff_history_len("YAGO", "offline")
    # run_diff_history_len("ICEWS18", "offline")
    run_diff_history_len("WIKI", "offline")
    # run_diff_history_len("ICEWS14", "offline")