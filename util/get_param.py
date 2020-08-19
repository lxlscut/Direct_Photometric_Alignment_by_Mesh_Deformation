import yaml

def get_param(parameter_name):
    f = open("./config.yaml", 'r', encoding='utf-8')
    cont = f.read()
    config = yaml.safe_load(cont)
    param = config.get(parameter_name)
    if param == None:
        print("there is no param with this name")
        return
    return param


# def set_param(parameter_name,value):
#     f = open("./config.yaml", 'wr', encoding='utf-8')
#     cont = f.read()
#     config = yaml.safe_load(cont)
#     param = config.setdefault()
#     if param == None:
#         print("there is no param with this name")
#         return
#     return param