import os, sys, torch, argparse, math, yaml
import numpy as np
from copy import deepcopy
from tqdm import tqdm, trange

sys.path.append('/data1/yang_liu/python_workspace/GSPL')

from internal.utils.vq import VectorQuantize
from internal.utils.gaussian_utils import GaussianPlyUtils
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.utils.vq_utils import read_ply_data, write_ply_data, load_vqgaussian

def parse_args():
    parser = argparse.ArgumentParser(description="vectree quantization")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)  
    parser.add_argument("--no_load_data", action='store_true')
    parser.add_argument("--no_save_ply", action='store_true', help="if true, save as .ckpt")
    parser.add_argument("--skip_quantize", action='store_true')
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--gs_dim", type=int, default=2, help="3 for 3DGS, 2 for 2DGS")

    parser.add_argument("--iteration_num", type=float, default=1000)
    parser.add_argument("--vq_ratio", type=float, default=0.4)
    parser.add_argument("--codebook_size", type=int, default=2**13)  # 2**13 = 8192
    parser.add_argument("--vq_way", type=str, default='half') 
    opt = parser.parse_args() 
    return opt
    

class Quantization():
    def __init__(self, opt):
        
        # ----- load model from ckpt or ply file -----
        if opt.sh_degree == 3:
            self.sh_dim = 3+45
        elif opt.sh_degree == 2:
            self.sh_dim = 3+24
        self.gs_dim = opt.gs_dim

        self.model_path = GaussianModelLoader.search_load_file(opt.model_path)
        ckpt = torch.load(self.model_path, map_location="cpu")
        # initialize modeld
        model = GaussianModelLoader.initialize_model_from_checkpoint(
            ckpt,
            device=device,
        )
        structred_model = GaussianPlyUtils.load_from_model(model).to_ply_format()
        model.freeze()
        model.pre_activate_all_properties()
        # initialize renderer
        renderer = GaussianModelLoader.initialize_renderer_from_checkpoint(
            ckpt,
            stage="validate",
            device=device,
        )
        try:
            dataparser_config = ckpt["datamodule_hyper_parameters"]["parser"]
        except:
            pass

        dataset_path = ckpt["datamodule_hyper_parameters"]["path"]

        dataparser_outputs = dataparser_config.instantiate(
            path=dataset_path,
            output_path=os.getcwd(),
            global_rank=0,
        ).get_outputs()

        # ----- get trim-gs style importance score -----
        if os.path.exists(f'{opt.save_path}/imp_score.npz'):
            self.importance = self.load_f(opt.save_path, 'imp_score.npz', allow_pickle=True)
        else:
            top_list = [None, ] * renderer.K
            with torch.no_grad():
                bg_color=torch.tensor(ckpt["hyper_parameters"]["background_color"], dtype=torch.float, device=device)
                for i in tqdm(range(len(dataparser_outputs.train_set.cameras)), desc="Calculating Importance Score"):
                    camera = dataparser_outputs.train_set.cameras[i].to_device(device)
                    trans = renderer(
                        camera,
                        model,
                        bg_color=bg_color,
                        record_transmittance=True
                    )
                    if top_list[0] is not None:
                        m = trans > top_list[0]
                        if m.any():
                            for i in range(renderer.K - 1):
                                top_list[renderer.K - 1 - i][m] = top_list[renderer.K - 2 - i][m]
                                top_list[0][m] = trans[m]
                    else:
                        top_list = [trans.clone() for _ in range(renderer.K)]

                self.importance = torch.stack(top_list, dim=-1).mean(-1).cpu().numpy()
                
                
        # tile = torch.quantile(contribution, self.prune_ratio)
        self.feats = np.concatenate([
            structred_model.xyz,
            np.zeros_like(structred_model.xyz),
            structred_model.features_dc.reshape((structred_model.features_dc.shape[0], -1)),
            structred_model.features_rest.reshape((structred_model.features_rest.shape[0], -1)),
            structred_model.opacities,
            structred_model.scales,
            structred_model.rotations
        ], axis=-1)
        del model, structred_model, dataparser_outputs
        self.feats = torch.tensor(self.feats)
        self.feats_bak = self.feats.clone()
        self.feats = self.feats[:, 6:6+self.sh_dim]

        # ----- define model -----
        self.model_vq = VectorQuantize(
                    dim = self.feats.shape[1],              
                    codebook_size = opt.codebook_size,
                    decay = 0.8,                            
                    commitment_weight = 1.0,                
                    use_cosine_sim = False,
                    threshold_ema_dead_code=0,
                ).to(device)
        
        # ----- other -----
        self.save_path = opt.save_path
        self.ply_path = opt.save_path
        self.high = None
        self.VQ_CHUNK = 80000
        self.k_expire = 10        
        self.vq_ratio = opt.vq_ratio

        self.no_load_data = opt.no_load_data
        self.no_save_ply = opt.no_save_ply
   
        self.codebook_size = opt.codebook_size
        self.iteration_num = opt.iteration_num
        self.vq_way = opt.vq_way

        # ----- print info -----
        print("\n================== Print Info ================== ")
        print("Input_feats_shape: ", self.feats_bak.shape)
        print("VQ_feats_shape: ", self.feats.shape)
        print("SH_degree: ", opt.sh_degree)
        print("Quantization_ratio: ", opt.vq_ratio)
        print("Add_important_score: ", False)
        print("Codebook_size: ", opt.codebook_size)
        print("================================================ ")

    @torch.no_grad()
    def calc_vector_quantized_feature(self):
        """
        apply vector quantize on gaussian attributes and return vq indexes
        """
        CHUNK = 8192
        feat_list = []
        indice_list = []
        self.model_vq.eval()
        self.model_vq._codebook.embed.half().float()   #
        for i in tqdm(range(0, self.feats.shape[0], CHUNK)):
            feat, indices, commit = self.model_vq(self.feats[i:i+CHUNK,:].unsqueeze(0).to(device))
            indice_list.append(indices[0])
            feat_list.append(feat[0])
        self.model_vq.train()
        all_feat = torch.cat(feat_list).half().float()  # [num_elements, feats_dim]
        all_indice = torch.cat(indice_list)             # [num_elements, 1]
        return all_feat, all_indice


    @torch.no_grad()
    def fully_vq_reformat(self):  

        print("\n=============== Start vector quantize ===============")
        all_feat, all_indice = self.calc_vector_quantized_feature()

        if self.save_path is not None:
            save_path = self.save_path
            os.makedirs(f'{save_path}/extreme_saving', exist_ok=True)

            # ----- save basic info -----
            metadata = dict()
            metadata['input_pc_num'] = self.feats_bak.shape[0]  
            metadata['input_pc_dim'] = self.feats_bak.shape[1]  
            metadata['codebook_size'] = self.codebook_size
            metadata['codebook_dim'] = self.sh_dim
            np.savez_compressed(f'{save_path}/extreme_saving/metadata.npz', metadata=metadata)

            # ===================================================== save vq_SH =============================================
            # ----- save mapping_index (vq_index) -----
            def dec2bin(x, bits):
                mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
                return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()    
            # vq indice was saved in according to the bit length
            self.codebook_vq_index = all_indice[torch.logical_xor(self.all_one_mask,self.non_vq_mask)]                             # vq_index
            bin_indices = dec2bin(self.codebook_vq_index, int(math.log2(self.codebook_size))).bool().cpu().numpy()                 # mapping_index
            np.savez_compressed(f'{save_path}/extreme_saving/vq_indexs.npz',np.packbits(bin_indices.reshape(-1)))               
            
            # ----- save codebook -----                                           
            codebook = self.model_vq._codebook.embed.cpu().half().numpy().squeeze(0)                                                 
            np.savez_compressed(f'{save_path}/extreme_saving/codebook.npz', codebook)

            # ----- save keep mask (non_vq_feats_index)-----
            np.savez_compressed(f'{save_path}/extreme_saving/non_vq_mask.npz',np.packbits(self.non_vq_mask.reshape(-1).cpu().numpy()))

            # ===================================================== save non_vq_SH =============================================
            non_vq_feats = self.feats_bak[self.non_vq_mask, 6:6+self.sh_dim]       
            wage_non_vq_feats = self.wage_vq(non_vq_feats)
            np.savez_compressed(f'{save_path}/extreme_saving/non_vq_feats.npz', wage_non_vq_feats) 

            # =========================================== save xyz & other attr(opacity + 3*scale + 4*rot) ====================================
            other_attribute = self.feats_bak[:, -8:]
            wage_other_attribute = self.wage_vq(other_attribute)
            np.savez_compressed(f'{save_path}/extreme_saving/other_attribute.npz', wage_other_attribute)

            xyz = self.feats_bak[:, 0:3]
            np.savez_compressed(f'{save_path}/extreme_saving/xyz.npz', xyz)  
            
            # =========================================== save importance score ====================================
            np.savez(os.path.join(opt.save_path, "imp_score"), self.importance)

        # zip everything together to get final size
        os.system(f"zip -r {save_path}/extreme_saving.zip {save_path}/extreme_saving")
        size = os.path.getsize(f'{save_path}/extreme_saving.zip')
        size_MB = size / 1024.0 / 1024.0
        print("Size = {:.2f} MB".format(size_MB))
            
        return all_feat, all_indice
    
    def load_f(self, path, name, allow_pickle=False,array_name='arr_0'):
        return np.load(os.path.join(path, name), allow_pickle=allow_pickle)[array_name]

    def wage_vq(self, feats):
        if self.vq_way == 'half':        
            return feats.half()
        else:
            return feats
    
    def quantize(self):

        ###################################################
        only_vq_some_vector = True
        if only_vq_some_vector:
            tensor_importance = torch.tensor(self.importance)
            large_val, large_index = torch.topk(tensor_importance, k=int(tensor_importance.shape[0] * (1-self.vq_ratio)), largest=True) 
            self.all_one_mask = torch.ones_like(tensor_importance).bool()     
            self.non_vq_mask = torch.zeros_like(tensor_importance).bool()         
            self.non_vq_mask[large_index] = True                         
        self.non_vq_index = large_index

        IS_non_vq_point = large_val.sum()
        IS_all_point = tensor_importance.sum()
        IS_percent = IS_non_vq_point/IS_all_point
        print("IS_percent: ", IS_percent)

        #=================== Codebook initialization & Update codebook ====================
        self.model_vq.train()
        with torch.no_grad():
            self.vq_mask = torch.logical_xor(self.all_one_mask, self.non_vq_mask)                  
            feats_needs_vq = self.feats[self.vq_mask].clone()                                       
            imp = tensor_importance[self.vq_mask].float()                                        
            k = self.k_expire                                                              
            if k > self.model_vq.codebook_size:
                k = 0            
            for i in trange(self.iteration_num):
                indexes = torch.randint(low=0, high=feats_needs_vq.shape[0], size=[self.VQ_CHUNK])         
                vq_weight = imp[indexes].to(device)
                vq_feature = feats_needs_vq[indexes,:].to(device)
                quantize, embed, loss = self.model_vq(vq_feature.unsqueeze(0), weight=vq_weight.reshape(1,-1,1))

                replace_val, replace_index = torch.topk(self.model_vq._codebook.cluster_size, k=k, largest=False)      
                _, most_important_index = torch.topk(vq_weight, k=k, largest=True)
                self.model_vq._codebook.embed[:,replace_index,:] = vq_feature[most_important_index,:]

        #=================== Apply vector quantization ====================
        all_feat, all_indices = self.fully_vq_reformat()

    def dequantize(self):
        print("\n==================== Load saved data & Dequantize ==================== ")
        dequantized_feats = load_vqgaussian(os.path.join(self.save_path,'extreme_saving'), device=device)

        if self.no_save_ply == False:
            os.makedirs(f'{self.ply_path}/', exist_ok=True)
            write_ply_data(dequantized_feats.cpu().numpy(), self.ply_path, self.sh_dim, self.gs_dim)
        else:
            checkpoint = torch.load(self.model_path)
            checkpoint["state_dict"]["gaussian_model.gaussians.means"] = dequantized_feats[:, :3]
            checkpoint["state_dict"]["gaussian_model.gaussians.shs_dc"] = dequantized_feats[:, 6:9].reshape(dequantized_feats.shape[0], 3, 1).permute(0, 2, 1)
            checkpoint["state_dict"]["gaussian_model.gaussians.shs_rest"] = dequantized_feats[:, 9:6+self.sh_dim].reshape(dequantized_feats.shape[0], 3, -1).permute(0, 2, 1)
            checkpoint["state_dict"]["gaussian_model.gaussians.opacities"] = dequantized_feats[:, 6+self.sh_dim:7+self.sh_dim]
            checkpoint["state_dict"]["gaussian_model.gaussians.scales"] = dequantized_feats[:, 7+self.sh_dim:-4]
            checkpoint["state_dict"]["gaussian_model.gaussians.rotations"] = dequantized_feats[:, -4:]
            torch.save(checkpoint, self.model_path.replace('.ckpt', '_vq.ckpt'))

if __name__=='__main__':
    opt = parse_args()
    device = torch.device('cuda')
    vq = Quantization(opt)

    if not opt.skip_quantize:
        vq.quantize()
    vq.dequantize()
    
    print("All done!")

