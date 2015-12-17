import json
import configparser
import numpy as np
import ast

class config_pact:

    def __init__(self, input_file='cross_match.ini'):


        conf = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        conf.read(input_file)


        ##################################################
        inputs = conf['inputs']

        self.input_cat = inputs['input_cat']
        
        ##################################################
        cat_sz = conf['sz_cat']

        self.szdb = cat_sz['sz_file']
        self.sz_fields = cat_sz['sz_fields'].split()
        self.sz_dist_theshold_arcmin = cat_sz.getfloat('sz_dist_theshold_arcmin')


         ##################################################
        cat_x = conf['x_cat']

        self.pxcc = cat_x['x_file']
        self.x_fields = cat_x['x_fields'].split()
        self.x_dist_theshold_arcmin = cat_x.getfloat('x_dist_theshold_arcmin')
	

        ##################################################
        cat_op = conf['op_cat']

        self.wen = cat_op['wen_file']
        self.wen_fields = cat_op['wen_fields'].split()
        self.wen_dist_theshold_arcmin = cat_op.getfloat('wen_dist_theshold_arcmin')
       
        self.rdm = cat_op['rdm_file']
        self.rdm_fields = cat_op['rdm_fields'].split()
        self.rdm_dist_theshold_arcmin = cat_op.getfloat('rdm_dist_theshold_arcmin')
       
        self.abell = cat_op['abell_file']
        self.abell_fields = cat_op['abell_fields'].split


        ##################################################
        ned = conf['ned']

        
        self.ned_radius_search = ned.getfloat('ned_radius_search')
        self.ned_fields = ast.literal_eval(conf.get('ned', "ned_fields"))
        self.ned_fields_format = ast.literal_eval(conf.get('ned', "ned_fields_format"))
        
        
        ##################################################
        outputs = conf['outputs']

        self.verbose = outputs.getboolean('verbose')
        self.outputs_filename = outputs['outputs_filename']
 


