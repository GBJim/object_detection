from subprocess import call


output_path = "example_output/"
config_path = "config/example_ssd_mobile.config"


CMD = ["python", "train.py", "--logtostderr", "--train_dir={}".format(output_path), "--pipeline_config_path={}".format(config_path) ] 
print(" ".join(CMD))


call(CMD)