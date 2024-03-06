from typing import Dict

from ..transformer_base import TRACK_TRANSFORMERS, TransformerBase
from videoanalyst.data.utils.target_image_crop import crop


@TRACK_TRANSFORMERS.register
class RandomCropTransformer(TransformerBase):

    default_hyper_params = dict(
        max_scale=0.3,
        max_shift=0.4,
        q_size=289,
        num_memory_frames=0,
        search_area_factor=0.0,
        phase_mode="train",
    )

    def __init__(self, seed: int = 0) -> None:
        super(RandomCropTransformer, self).__init__(seed=seed)

    def __call__(self, sampled_data: Dict) -> Dict:
        data1 = sampled_data["data1"]
        data2 = sampled_data["data2"]
        sampled_data["data1"] = {}
        nmf = self._hyper_params['num_memory_frames']
        search_area_factor = self._hyper_params['search_area_factor']
        q_size = self._hyper_params['q_size']
        im_memory_hsi = data1["hsi_img"]
        im_memory_material = data1["material_img"]
        im_memory_fc = data1["fc_img"]
        bbox_memory = data1["anno"]
        im_memory_hsi, _, _ = crop(im_memory_hsi, bbox_memory, search_area_factor, q_size,
                                config=self._hyper_params, rng=self._state["rng"])
        im_memory_material, _, _ = crop(im_memory_material, bbox_memory, search_area_factor, q_size,
                                config=self._hyper_params, rng=self._state["rng"])
        im_memory_fc, bbox_m, _ = crop(im_memory_fc, bbox_memory, search_area_factor, q_size,
                                config=self._hyper_params, rng=self._state["rng"])
        sampled_data["data1"] = dict(hsi_img=im_memory_hsi, material_img=im_memory_material, fc_img=im_memory_fc, anno=bbox_m)

        im_query_hsi = data2["hsi_img"]
        im_query_material = data2["material_img"]
        im_query_fc = data2["fc_img"]
        bbox_query = data2["anno"]

        im_q_hsi, _, _ = crop(im_query_hsi, bbox_query, search_area_factor, q_size,
                               config=self._hyper_params, rng=self._state["rng"])
        im_q_material, _, _ = crop(im_query_material, bbox_query, search_area_factor, q_size,
                               config=self._hyper_params, rng=self._state["rng"])
        im_q_fc, bbox_q, _ = crop(im_query_fc, bbox_query, search_area_factor, q_size,
                               config=self._hyper_params, rng=self._state["rng"])
        sampled_data["data2"] = dict(hsi_img=im_q_hsi, material_img=im_q_material, fc_img=im_q_fc, anno=bbox_q)

        return sampled_data
