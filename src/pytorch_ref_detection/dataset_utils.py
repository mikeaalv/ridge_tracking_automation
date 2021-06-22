import copy
import os
from PIL import Image
import pandas as pd
import json
import numpy as np
from PIL import Image

import torch
import torch.utils.data
import torchvision

from pycocotools import mask as coco_mask
from pycocotools.coco import COCO

import transforms as T


# class FilterAndRemapCocoCategories(object):
#     def __init__(self,categories,remap=True):
#         self.categories=categories
#         self.remap=remap
#
#     def __call__(self,image,target):
#         anno=target["annotations"]
#         anno=[obj for obj in anno if obj["category_id"] in self.categories]
#         if not self.remap:
#             target["annotations"]=anno
#             return image,target
#         anno=copy.deepcopy(anno)
#         for obj in anno:
#             obj["category_id"]=self.categories.index(obj["category_id"])
#         target["annotations"]=anno
#         return image,target

def convert_coco_poly_to_mask(segmentations,height,width):
    masks=[]
    for polygons in segmentations:
        rles=coco_mask.frPyObjects(polygons,height,width)
        mask=coco_mask.decode(rles)
        if len(mask.shape)<3:
            mask=mask[..., None]
        mask=torch.as_tensor(mask,dtype=torch.uint8)
        mask=mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks=torch.stack(masks,dim=0)
    else:
        masks=torch.zeros((0,height, width),dtype=torch.uint8)
    return masks

class ConvertCocoPolysToMask(object):
    def __call__(self,image,target):
        w,h=image.size
        image_id=target["image_id"]
        image_id=torch.tensor([image_id])
        anno=target["annotations"]
        anno=[obj for obj in anno if obj['iscrowd'] == 0]
        boxes=[obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes=torch.as_tensor(boxes,dtype=torch.float32).reshape(-1,4)
        boxes[:,2:] += boxes[:,:2]
        boxes[:,0::2].clamp_(min=0,max=w)
        boxes[:,1::2].clamp_(min=0,max=h)
        classes=[obj["category_id"] for obj in anno]
        classes=torch.tensor(classes,dtype=torch.int64)
        segmentations=[obj["segmentation"] for obj in anno]
        masks=convert_coco_poly_to_mask(segmentations, h, w)
        keypoints=None
        if anno and "keypoints" in anno[0]:
            keypoints=[obj["keypoints"] for obj in anno]
            keypoints=torch.as_tensor(keypoints,dtype=torch.float32)
            num_keypoints=keypoints.shape[0]
            if num_keypoints:
                keypoints=keypoints.view(num_keypoints,-1,3)
        #
        keep=(boxes[:,3]>boxes[:,1]) & (boxes[:,2]>boxes[:,0])
        boxes=boxes[keep]
        classes=classes[keep]
        masks=masks[keep]
        if keypoints is not None:
            keypoints=keypoints[keep]
        #
        target={}
        target["boxes"]=boxes
        target["labels"]=classes
        target["masks"]=masks
        target["image_id"]=image_id
        if keypoints is not None:
            target["keypoints"]=keypoints
        # for conversion to coco api
        area=torch.tensor([obj["area"] for obj in anno])
        iscrowd=torch.tensor([obj["iscrowd"] for obj in anno])
        target["area"]=area
        target["iscrowd"]=iscrowd
        #
        return image, target

# def _coco_remove_images_without_annotations(dataset,cat_list=None):
#     def _has_only_empty_bbox(anno):
#         return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)
#     #
#     def _count_visible_keypoints(anno):
#         return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)
#     #
#     min_keypoints_per_image=10
#     def _has_valid_annotation(anno):
#         # if it's empty, there is no annotation
#         if len(anno) == 0:
#             return False
#         # if all boxes have close to zero area, there is no annotation
#         if _has_only_empty_bbox(anno):
#             return False
#         # keypoints task have a slight different critera for considering
#         # if an annotation is valid
#         if "keypoints" not in anno[0]:
#             return True
#         # for keypoint detection tasks, only consider valid images those
#         # containing at least min_keypoints_per_image
#         if _count_visible_keypoints(anno) >= min_keypoints_per_image:
#             return True
#         return False
#     #
#     # assert isinstance(dataset,torchvision.datasets.CocoDetection)
#     ids=[]
#     for ds_idx, img_id in enumerate(dataset.ids):
#         ann_ids=dataset.coco.getAnnIds(imgIds=img_id,iscrowd=None)
#         anno=dataset.coco.loadAnns(ann_ids)
#         if cat_list:
#             anno=[obj for obj in anno if obj["category_id"] in cat_list]
#         if _has_valid_annotation(anno):
#             ids.append(ds_idx)
    #
    # dataset=torch.utils.data.Subset(dataset,ids)
    # return dataset


def convert_to_coco_api(ds):
    # for engine.py
    coco_ds=COCO()
    # annotation IDs need to start at 1, not 0, see torchvision issue #1530
    ann_id=1
    dataset={'images':[],'categories':[],'annotations':[]}
    categories=set()
    for img_idx in range(len(ds)):
        # find better way to get target
        img,targets=ds[img_idx]
        image_id=targets["image_id"].item()
        img_dict={}
        img_dict['id']=image_id
        img_dict['height']=img.shape[-2]
        img_dict['width']=img.shape[-1]
        dataset['images'].append(img_dict)
        bboxes=targets["boxes"]
        bboxes[:,2:] -= bboxes[:,:2]
        bboxes=bboxes.tolist()
        labels=targets['labels'].tolist()
        areas=targets['area'].tolist()
        iscrowd=targets['iscrowd'].tolist()
        if 'masks' in targets:
            masks=targets['masks']
            # make masks Fortran contiguous for coco_mask
            masks=masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
        if 'keypoints' in targets:
            keypoints=targets['keypoints']
            keypoints=keypoints.reshape(keypoints.shape[0],-1).tolist()
        num_objs=len(bboxes)
        for i in range(num_objs):
            ann={}
            ann['image_id']=image_id
            ann['bbox']=bboxes[i]
            ann['category_id']=labels[i]
            categories.add(labels[i])
            ann['area']=areas[i]
            ann['iscrowd']=iscrowd[i]
            ann['id']=ann_id
            if 'masks' in targets:
                ann["segmentation"]=coco_mask.encode(masks[i].numpy())
            if 'keypoints' in targets:
                ann['keypoints']=keypoints[i]
                ann['num_keypoints']=sum(k != 0 for k in keypoints[i][2::3])
            dataset['annotations'].append(ann)
            ann_id += 1
    #
    dataset['categories']=[{'id': i} for i in sorted(categories)]
    coco_ds.dataset=dataset
    coco_ds.createIndex()
    return coco_ds


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        if isinstance(dataset, torchvision.datasets.CocoDetection):
            break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset=dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco
    return convert_to_coco_api(dataset)


# class CocoDetection(torchvision.datasets.CocoDetection):
#     def __init__(self,img_folder,ann_file,transforms):
#         super(CocoDetection,self).__init__(img_folder,ann_file)
#         self._transforms=transforms
#
#     def __getitem__(self,idx):
#         img,target=super(CocoDetection,self).__getitem__(idx)
#         image_id = self.ids[idx]
#         target = dict(image_id=image_id, annotations=target)
#         if self._transforms is not None:
#             img, target = self._transforms(img, target)
#         return img, target

class loaddataset(torch.utils.data.Dataset):
    # modified from PennFudanDataset at https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
    def __init__(self,root,transforms,classes):
        self.root=root
        self.transforms=transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs=list(sorted(os.listdir(os.path.join(root))))
        anno_file=os.path.join(self.root,"regiondata.csv")
        self.annotab=pd.read_csv(anno_file,delimiter="\t")
        self.ids=[i for i in range(len(self.imgs))]
        self.classes=classes
    
    def __getitem__(self,idx):
        # load images
        file=self.imgs[idx]
        annotab=self.annotab
        img_path=os.path.join(self.root,file)
        img=Image.open(img_path).convert("RGB")
        subtab=annotab[annotab['filename']==file]
        num_objs=subtab.shape[0]
        # tab_rec=subtab.iloc[anno_i]
        # assert not tab_rec["region_attributes"]#check it is []
        # anno=json.loads(tab_rec["region_shape_attributes"])
        # labelclass=tab_rec["region_attibutes_named"]
        boxes=[]
        segmentations=[]
        for anno_i in range(num_objs):#multiple masks/boxes
            tab_rec=subtab.iloc[anno_i]
            # assert not tab_rec["region_attributes"]#check it is []
            anno=json.loads(tab_rec["region_shape_attributes"])
            labelclass=tab_rec["region_attibutes_named"]
            if len(anno)==0:
                continue
            #
            px=anno["all_points_x"]
            py=anno["all_points_y"]
            poly=[(x+0.5,y+0.5) for x,y in zip(px,py)]
            poly=[p for x in poly for p in x]
            category=np.where([ele==labelclass for ele in self.classes])[0]
            boxes.append([np.min(px),np.min(py),np.max(px),np.max(py)])
            segmentations.append([poly])
        #
        # convert everything into a torch.Tensor
        boxes=torch.as_tensor(boxes,dtype=torch.float32)
        # there is only one class
        labels=torch.ones((num_objs,),dtype=torch.int64)
        image_id=torch.tensor([idx])
        area=(boxes[:,3]-boxes[:,1])*(boxes[:,2]-boxes[:,0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,),dtype=torch.int64)
        target={}
        target["boxes"]=boxes
        target["labels"]=labels
        target["segmentation"]=segmentations
        target["image_id"]=image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        #
        if self.transforms is not None:
            img,target=self.transforms(img,target)
        #
        return img,target
        
    def __len__(self):
        return len(self.imgs)

def get_dataset_loc(root,image_set,transforms,classes,mode='instances'):
    anno_file_template="regiondata.csv"
    PATHS = {
        "train": ("train",os.path.join("train/",anno_file_template)),
        "val": ("validate",os.path.join("train/",anno_file_template)),
    }
    # add the mask converter to the end of augmentation/transformation
    t=[ConvertCocoPolysToMask()]
    if transforms is not None:
        t.append(transforms)
    transforms=T.Compose(t)
    #
    img_folder,ann_file=PATHS[image_set]
    img_folder=os.path.join(root,img_folder)
    # load data
    dataset=loaddataset(img_folder,transforms=transforms,classes=classes)
    #
    # if image_set=="train":
    #     dataset=_coco_remove_images_without_annotations(dataset)
    #
    return dataset
