'''
Get request paramters from Flask.request
'''

from typing import List, Type, Union, Dict, Any

def chk_req(p:Dict[str,Any], req_p:Union[List[str], set]) -> Union[str, None]:

    '''
    Check that the required parameters have been passed into the request. 

    Inputs:
        - p : Dict[str:Any] -- dict holding all parameters read from the request
        - req_p : list or set -- list of required parameters (value of these parameters must be passed in request)
    '''

    missing_p = []

    for name in req_p:
        if name in p:
            val = p[name]
        else:
            return f"Error with parameters dictionary p: key {name} not found."
        if val == None:  # if parameter is missing from the request
            missing_p.append(name)

    if len(missing_p) != 0:
        return f"Input Parameter Error: {missing_p}. Parameter(s) Not Found.\nThe required parameters listed were not found in the submitted request."


def extract(req_args:"flask.request.args", para_names:Union[List[str], set]) -> Dict[str, Any]:

    '''
    Get parameter values from request args and return a dictionary of the parameters.

    Inputs:
        - req_args : flask.request.args -- request arguments from flask request
        - para_names : list or set -- names of parameters to extract from request
    '''
    
    p = {}
    for name in para_names:
        p[name] = req_args.get(name)
    return p
    

def set_def(p:Dict[str, Any], def_p:Dict[str, Any]) -> Dict[str, Any]:
    
    '''
    Set default values for the parameters.

    Inputs:
        - p : Dict[str:Any] -- dict holding all parameters read from the request
        - def_p : Dict[str:Any] -- dict holding the optional parameter names and their default values
    '''
    
    for opt_var, def_val in def_p.items():
        if opt_var in p:
            if p[opt_var] == None:
                p[opt_var] = def_val
        else:
            print(f"Error setting value of optional parameter {opt_var} to {def_val}. Value not found in parameters dict.")

    return p


def get_para(req_args: "flask.request.args", criteria:dict) -> Dict[str, Any]:
    
    '''
    Combines the extraction, checking for required parameters and setting default values

    Inputs:
        - req_args: flask.request.args object -- request arguments from flask request
        - criteria: dict -- dict specifying the parameters to extract, required parameters and default values.
            - format : {'req':[required_str_names], 'opt': {optional_str_names:default_values}}
    '''

    # required parameters
    req_para_names = criteria['req']
    req_p = extract(req_args, req_para_names)
    chk_req(req_p, req_para_names)

    # optional parameters
    opt_para_info = criteria['opt']
    opt_p = extract(req_args, opt_para_info.keys())
    opt_p = set_def(opt_p, opt_para_info)

    return {**req_p, **opt_p}

def eval_args(args, to_eval):
    for name, val in args.items():
        if name in to_eval:
            if isinstance(args[name], str):
                args[name] = eval(val)
    return args