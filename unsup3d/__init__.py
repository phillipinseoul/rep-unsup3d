'''
will be used to initialize...
'''
is_debug = False
use_sched = True
EPS = 1e-7
test_ELU = False
torch_old = True
USE_WIDER_DEPTH = False
USE_GRADIENT_CLIP = True


# Options for ablation

WITH_ALBEDO_FLIP = True #done
WITH_DEPTH_FLIP = True  #done
WITH_LIGHT = True   # predict shading map directly ()
WITH_PERCEP = True  # done
WITH_CONF = True    # done

WITH_SELF_SUP_PERCEP = False    # done
WITH_GT_DEPTH = False
WITH_PERTURB = False