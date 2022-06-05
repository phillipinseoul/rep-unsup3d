'''
will be used to initialize...
'''
# ignorable settings
is_debug = False
USE_SCHED = False
EPS = 1e-7
test_ELU = False
torch_old = True # debug option (default is true)
USE_WIDER_DEPTH = False

# options for SIDE, MAD baselines
test_supervised = False
USE_GRADIENT_CLIP = True
VISUALIZE_RESULTS = True

'''
# important settings
USE_GRADIENT_CLIP = True


# Options for ablation

WITH_ALBEDO_FLIP = True #done
WITH_DEPTH_FLIP = True  #done
WITH_LIGHT = True   # predict shading map directly ()
WITH_PERCEP = True  # done
WITH_CONF = False    # done

WITH_SELF_SUP_PERCEP = False    # done
WITH_GT_DEPTH = False
WITH_PERTURB = False
'''

'''

def init_settings(configs):
    global USE_GRADIENT_CLIP
    global WITH_ALBEDO_FLIP
    global WITH_DEPTH_FLIP
    global WITH_LIGHT
    global WITH_PERCEP
    global WITH_CONF
    global WITH_SELF_SUP_PERCEP
    global WITH_GT_DEPTH
    global WITH_PERTURB


    USE_GRADIENT_CLIP = configs.get('gradient_clip', True)

    WITH_ALBEDO_FLIP = configs.get('with_abledo_flip', True) #done
    WITH_DEPTH_FLIP = configs.get('with_depth_flip', True)  #done
    WITH_LIGHT = configs.get('with_light', True)   # predict shading map directly ()
    WITH_PERCEP = configs.get('with_percep', True)  # done
    WITH_CONF = configs.get('with_conf', True)    # done

    WITH_SELF_SUP_PERCEP = configs.get('with_self_sup_percep', False)    # done
    WITH_GT_DEPTH = configs.get('with_gt_depth', False)
    WITH_PERTURB = configs.get('with_perturb', False)
'''