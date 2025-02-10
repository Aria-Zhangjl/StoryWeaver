from packaging import version
from PIL import Image
from torchvision import transforms
import os
import PIL
from torch.utils.data import Dataset
import torchvision
import numpy as np
import torch
import random
import albumentations as A
import copy
import cv2
import pandas as pd
from glob import glob
import re

class TBC_Bench_Single_Story(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        resolution=512,
        max_char = 2,
        device='cuda', 
        caption_dir='',
        ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.frame_texts = []
        with open(data_root, 'r') as f:
            for line in f:
                self.frame_texts.append(line.strip())
        
        self.caption_dir = glob(os.path.join(caption_dir, '*.txt'))
        dir_name = os.path.dirname(caption_dir)
        style_file = os.path.join(dir_name, 'style.txt')

        self.styletext = self.obtain_style(style_file)
        self.obtain_character_caption()
        self.max_char = max_char
        self.num_images = len(self.frame_texts)
        print('Total {} images ...'.format(self.num_images))

    def obtain_style(self, style_file):
        with open(style_file, 'r') as f:
            text = f.readlines()
        text = text[0].strip()
        return text

    def __len__(self):
        return self.num_images

    def obtain_character_caption(self,):
        self.character_desc_ids = {}
        self.character_desc_caption = {}
        self.char_class_ids = {}
        for each_txt in self.caption_dir:
            with open(each_txt, 'r') as f:
                text = f.readlines()
            text = text[0].strip()
            char_name = os.path.basename(each_txt).split('.')[0]
            self.character_desc_ids[char_name] = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids[0]
            self.character_desc_caption[char_name] = text
            self.char_class_ids[char_name] = self.tokenizer(
                self.char_class[char_name],
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids[0]

    def get_full_text(self, text):
        text = text.replace(f'<style>', self.styletext)
        for each_char in self.character_desc_caption.keys():
            text = text.replace(f'<{each_char}>', self.character_desc_caption[each_char])
        return text

    def obtain_text(self, text):
        character_desc_index = []
        character_list = []
        text = self.get_full_text(text)
        input_ids = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        for each_char in self.character_desc_ids.keys():
            window_size = len(self.character_desc_ids[each_char][1:8])
            character_start_indice = torch.where(torch.all(input_ids.unfold(0, window_size, 1) == self.character_desc_ids[each_char][1:8], dim=1))[0]
            if len(character_start_indice) > 0: 
                character_list.append(self.character_desc_ids[each_char])
                for cnt, each in enumerate(self.character_desc_ids[each_char]):
                    token = self.tokenizer.decoder[each.item()]
                    if token == self.tokenizer.eos_token:
                        length = cnt - 1
                        break
                character_end_indice = character_start_indice + length
                window_size = len(self.char_class_ids[each_char][1:2])
                char_class_start_indice = torch.where(torch.all(input_ids.unfold(0, window_size, 1) == self.char_class_ids[each_char][1:2], dim=1))[0][0]
                for cnt, each in enumerate(self.char_class_ids[each_char]):
                    # print(each_char,self.char_class_ids[each_char])
                    token = self.tokenizer.decoder[each.item()]
                    if token == self.tokenizer.eos_token:
                        length = cnt - 1
                        break
                char_class_end_indice = char_class_start_indice + length
                # print([character_start_indice.item(), character_end_indice.item(),char_class_start_indice.item(), char_class_end_indice.item()])
                character_desc_index.append([character_start_indice.item(), character_end_indice.item(),char_class_start_indice.item(), char_class_end_indice.item()])
        sorted_list = sorted(zip(character_desc_index, character_list))
        character_desc_index, character_list = zip(*sorted_list)
        character_desc_index = list(character_desc_index)
        character_list = list(character_list)
        _final_index = character_desc_index[-1]
        _final_ids = character_list[-1]
        self.multi_char = len(character_desc_index)
        while len(character_desc_index) != self.max_char:
            character_desc_index.append(_final_index)
            character_list.append(_final_ids)
        return input_ids, text, character_list, character_desc_index

    def obtain_prior_dist(self, total_char):
        var = torch.eye(2).unsqueeze(0).repeat(self.max_char,1,1)
        if total_char > 1:
            per_mean = torch.linspace(-1+(1/self.max_char), 1-(1/self.max_char), total_char)
        else:
            per_mean = torch.zeros(1)
        mean = [[0, i.item()] for i in per_mean]
        pad_len = self.max_char - len(mean)
        for i in range(pad_len):
            mean.append([0,0])
        mean = torch.tensor(mean)
        return var, mean

    def __getitem__(self, i):
        example = {}
        frame_text = self.frame_texts[i]
        input_ids, text, character_list, character_desc_index = self.obtain_text(frame_text)
        example["input_ids"] = input_ids
        example["text"] = text
        example["character_desc_index"] = character_desc_index
        example["character_list"] = character_list
        example['cor'], example['mean'] = self.obtain_prior_dist(self.multi_char)
        return example

class BenchMark_Each_ClassConstrains(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        resolution=512,
        center_crop=False,
        random_flip=True,
        max_char = 3,
        type='Pororo',
        ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.image_dir = glob(os.path.join(data_root, 'image/*/*.png'))
        self.character_dir = glob(os.path.join(data_root, 'character/*.txt'))
        self.num_images = len(self.image_dir)
        self.resolution = resolution
        if type == 'Pororo':
            self.char_class={
                'Crong':'dinosaur',
                'Eddy': 'fox',
                'Harry': 'bird',
                'Loopy': 'beaver',
                'Petty': 'Adélie penguin',
                'Poby':'polar bear',
                'Pororo':'gentoo penguin',
                'Rody':'robot',
                'Tongtong':'dragon',
            }
        elif type == 'Frozen':
            self.char_class={
                'Anna':'brown-hair girl',
                'Elsa': 'white-hair girl',
                'Kristoff': 'man',
                'Olaf': 'snowman',
                'Sven': 'reindeer',
            }            
        self.obtain_character_caption(self.character_dir)
        self.max_char = max_char

        self.style_file = os.path.join(data_root, 'style.txt')
        print('Total {} images ...'.format(self.num_images))

        self.transforms = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop((resolution,int(resolution / 9) * 16)) if center_crop else transforms.RandomCrop((resolution,int(resolution / 9) * 16)),
                transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )


    def __len__(self):
        return self.num_images

    def obtain_character_caption(self,character_dir):
        self.character_desc = {}
        self.character_desc_ids = {}
        self.char_class_ids = {}
        for each_txt in character_dir:
            with open(each_txt, 'r') as f:
                text = f.readlines()
            text = text[0].strip()
            char_name = os.path.basename(each_txt).split('.')[0]
            self.character_desc_ids[char_name] = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids[0]
            self.char_class_ids[char_name] = self.tokenizer(
                self.char_class[char_name],
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids[0]
            self.character_desc[char_name] = text

    def obtain_style(self, per_img_dir):
        with open(self.style_file, 'r') as f:
            text = f.readlines()
        text = text[0].strip()
        return text

    def get_full_text(self, text_file,text):
        style_txt = self.obtain_style(text_file)
        text = text.replace('<style>', style_txt)
        matches = re.findall(r"<(.*?)>", text)
        for each_char in matches:
            text = text.replace(f'<{each_char}>', self.character_desc[each_char])
        return text


    def obtain_text(self, text_file):
        character_desc_index = []
        character_list = []
        with open(text_file, 'r') as f:
            text = f.readlines()
        text = text[0].strip()
        text = self.get_full_text(text_file, text)
        input_ids = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        for each_char in self.character_desc_ids.keys():
            window_size = len(self.character_desc_ids[each_char][1:8])
            character_start_indice = torch.where(torch.all(input_ids.unfold(0, window_size, 1) == self.character_desc_ids[each_char][1:8], dim=1))[0]
            if len(character_start_indice) > 0: 
                character_list.append(self.character_desc_ids[each_char])
                for cnt, each in enumerate(self.character_desc_ids[each_char]):
                    token = self.tokenizer.decoder[each.item()]
                    if token == self.tokenizer.eos_token:
                        length = cnt - 1
                        break
                character_end_indice = character_start_indice + length
                window_size = len(self.char_class_ids[each_char][1:2])
                char_class_start_indice = torch.where(torch.all(input_ids.unfold(0, window_size, 1) == self.char_class_ids[each_char][1:2], dim=1))[0][0]
                for cnt, each in enumerate(self.char_class_ids[each_char]):
                    token = self.tokenizer.decoder[each.item()]
                    if token == self.tokenizer.eos_token:
                        length = cnt - 1
                        break
                char_class_end_indice = char_class_start_indice + length
                character_desc_index.append([character_start_indice.item(), character_end_indice.item(),char_class_start_indice.item(), char_class_end_indice.item()])
        
        
        sorted_list = sorted(zip(character_desc_index, character_list))
        character_desc_index, character_list = zip(*sorted_list)
        character_desc_index = list(character_desc_index)
        character_list = list(character_list)
        _final_index = character_desc_index[-1]
        _final_ids = character_list[-1]
        self.actual_char = len(character_desc_index)
        while len(character_desc_index) != self.max_char:
            character_desc_index.append(_final_index)
            character_list.append(_final_ids)
        return input_ids, text, character_list, character_desc_index

    def obtain_prior_dist(self, total_char):
        var = torch.eye(2).unsqueeze(0).repeat(self.max_char,1,1)
        if total_char > 1:
            per_mean = torch.linspace(-1+(1/self.max_char), 1-(1/self.max_char), total_char)
        else:
            per_mean = torch.zeros(1)
        mean = [[0, i.item()] for i in per_mean]
        pad_len = self.max_char - len(mean)
        for i in range(pad_len):
            mean.append([0,0])
        mean = torch.tensor(mean)
        return var, mean

    def __getitem__(self, i):
        example = {}
        image_file = self.image_dir[i]
        text_file = self.image_dir[i].replace('png', 'txt').replace('/image', '/caption')
        # print(text_file)
        input_ids, text, character_list, character_desc_index = self.obtain_text(text_file)
        example["input_ids"] = input_ids
        example["text"] = text
        example["character_desc_index"] = character_desc_index
        example["character_list"] = character_list
        try:
            img_train = Image.open(image_file).convert("RGB")
            example["pixel_values"] = self.transforms(img_train)
            
        except Exception as e:
            with open(self.error_text, "a") as file:
                file.write(image_file + "\n")
            example["pixel_values"] = torch.zeros((3, 512, 512))
            with open('error.txt', 'a+') as f:
                f.write(str(e) + '\n')
        example['cor'], example['mean'] = self.obtain_prior_dist(self.actual_char)
        return example

class Benchmark_Eval_ClassConstrains(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        resolution=512,
        max_char = 2,
        device='cuda', 
        type='Pororo',
        ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.caption_dir = glob(os.path.join(data_root, '*.txt'))
        dirname = os.path.dirname(data_root)
        self.char_cap_dir = glob(os.path.join(dirname, 'character', '*.txt'))
        if type == 'Pororo':
            self.char_class={
                'Crong':'dinosaur',
                'Eddy': 'fox',
                'Harry': 'bird',
                'Loopy': 'beaver',
                'Petty': 'Adélie penguin',
                'Poby':'polar bear',
                'Pororo':'gentoo penguin',
                'Rody':'robot',
                'Tongtong':'dragon',

            }
        elif type == 'Frozen':
            self.char_class={
                'Anna':'brown-hair girl',
                'Elsa': 'white-hair girl',
                'Kristoff': 'man',
                'Olaf': 'snowman',
                'Sven': 'reindeer',

            }
        self.obtain_character_caption(self.char_cap_dir)
        self.max_char = max_char
        self.num_images = len(self.caption_dir)
        print('Total {} images ...'.format(self.num_images))


    def __len__(self):
        return self.num_images

    def obtain_character_caption(self,character_dir):
        self.character_desc = {}
        self.character_desc_ids = {}
        self.char_class_ids = {}
        for each_txt in character_dir:
            with open(each_txt, 'r') as f:
                text = f.readlines()
            text = text[0].strip()
            char_name = os.path.basename(each_txt).split('.')[0]
            if char_name not in self.char_class.keys():
                continue
            self.character_desc_ids[char_name] = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids[0]
            self.character_desc[char_name] = text
            self.char_class_ids[char_name] = self.tokenizer(
                self.char_class[char_name],
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids[0]

    def get_full_text(self, text_file,text):
        matches = re.findall(r"<(.*?)>", text)
        for each_char in matches:
            text = text.replace(f'<{each_char}>', self.character_desc[each_char])
        return text

    def obtain_text(self, text_file):
        character_desc_index = []
        character_list = []
        with open(text_file, 'r') as f:
            text = f.readlines()
        text = text[0].strip()
        text = self.get_full_text(text_file, text)
        input_ids = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        for each_char in self.character_desc_ids.keys():
            window_size = len(self.character_desc_ids[each_char][1:8])
            character_start_indice = torch.where(torch.all(input_ids.unfold(0, window_size, 1) == self.character_desc_ids[each_char][1:8], dim=1))[0]
            if len(character_start_indice) > 0: 
                character_list.append(self.character_desc_ids[each_char])
                for cnt, each in enumerate(self.character_desc_ids[each_char]):
                    token = self.tokenizer.decoder[each.item()]
                    if token == self.tokenizer.eos_token:
                        length = cnt - 1
                        break
                character_end_indice = character_start_indice + length
                window_size = len(self.char_class_ids[each_char][1:2])
                char_class_start_indice = torch.where(torch.all(input_ids.unfold(0, window_size, 1) == self.char_class_ids[each_char][1:2], dim=1))[0][0]
                for cnt, each in enumerate(self.char_class_ids[each_char]):
                    token = self.tokenizer.decoder[each.item()]
                    if token == self.tokenizer.eos_token:
                        length = cnt - 1
                        break
                char_class_end_indice = char_class_start_indice + length
                character_desc_index.append([character_start_indice.item(), character_end_indice.item(),char_class_start_indice.item(), char_class_end_indice.item()])
        sorted_list = sorted(zip(character_desc_index, character_list))
        character_desc_index, character_list = zip(*sorted_list)
        character_desc_index = list(character_desc_index)
        character_list = list(character_list)
        _final_index = character_desc_index[-1]
        _final_ids = character_list[-1]
        self.actual_char = len(character_desc_index)
        while len(character_desc_index) != self.max_char:
            character_desc_index.append(_final_index)
            character_list.append(_final_ids)
        return input_ids, text, character_list, character_desc_index

    def obtain_prior_dist(self, total_char):
        var = torch.eye(2).unsqueeze(0).repeat(self.max_char,1,1)
        if total_char > 1:
            per_mean = torch.linspace(-1+(1/self.max_char), 1-(1/self.max_char), total_char)
        else:
            per_mean = torch.zeros(1)
        mean = [[0, i.item()] for i in per_mean]
        pad_len = self.max_char - len(mean)
        for i in range(pad_len):
            mean.append([0,0])
        mean = torch.tensor(mean)
        return var, mean

    def __getitem__(self, i):
        example = {}
        gen_caption_dir = self.caption_dir[i]
        input_ids, text, character_list, character_desc_index = self.obtain_text(gen_caption_dir)
        example["input_ids"] = input_ids
        example["text"] = text
        example["character_desc_index"] = character_desc_index
        example["character_list"] = character_list
        example['cor'], example['mean'] = self.obtain_prior_dist(self.actual_char)
        return example

