import glob as glob
import shutil

exp_alg = 'MAAC'
env_id = 'simple_line_po'
exp_desc = 'std'
seeds = [1,2001,4001,6001,8001]

colldata_dir = 'experiments/{}_multipleseeds_data_{}_{}_{}'.format(exp_alg,exp_alg,env_id, exp_desc)

for seed in seeds:
    try:
        # ss = 'experiments/*_{}_*_{}_{}_seed{}'.format(env_id,exp_alg,exp_desc,seed)
        ss = 'experiments/*_{}_{}_{}_{}_seed{}'.format(env_id,exp_alg,exp_alg,exp_desc,seed)
        print(ss)
        src_dir = glob.glob(ss)[0]
        print(src_dir)
        shutil.copyfile('{}/summary/reward_total.txt'.format(src_dir),
                        '{}/reward_training_seed{}.txt'.format(colldata_dir,seed)
                            )
        print('copied')                
    except Exception as e:
        print('Seed {} not copied'.format(seed))                
        print(e)                
        
