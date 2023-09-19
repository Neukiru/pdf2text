from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File,UploadFile,Depends,Query
from fastapi.staticfiles import StaticFiles
from PIL import Image
import numpy as np
from dataclasses import dataclass
from typing import List
from datetime import datetime
import os
import math
from random import random
from scipy import ndimage
import fitz
import numpy as np
from PIL import Image, ImageEnhance

import shutil


@dataclass
class ModelInterface:
    pdf_name: str = Query(...)

app = FastAPI()
origins = [ "*" ]




app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)



app.mount("/static", StaticFiles(directory="static"), name="static")

def pdf_get_text(pdf, folder_name):
    fitz.TOOLS.set_small_glyph_heights(False)
    pdf_name = pdf.split('.')[0]
    doc = fitz.open(folder_name + '/' + pdf_name + '.pdf')
    doc_w = doc[0].rect.round().width
    doc_h = doc[0].rect.round().height
    all_bboxes_list = []
    draw_bbox_list = []
    img_bbox_list = []
    span_bbox_list = []
    all_pages = []
    for pnum, page in enumerate(doc, 1):
        if page.rect.round().width > page.rect.round().height:
            page.set_rotation(-90)
        # <<<< Getting All Drawings BBOXes >>>>
        paths = page.get_cdrawings()
        drawings_array = np.zeros((doc_h, doc_w), dtype=np.uint8)
        for path in paths:
            r = fitz.Rect(path['rect'])
            bbox = fitz.IRect(r[0] - 0, r[1] - 0, r[2] + 1, r[3] + 1) * page.rotation_matrix
            drawings_array[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
        drawings_array, drawings_blob_rects = find_areas(drawings_array)
        draw_bbox_list.append(drawings_blob_rects)
        # save_arr_as_image(drawings_array * 255, str(pnum) + '_drawings.png')  # << debugging line

        # <<<< Getting All Spans and Images BBOXes >>>>
        span_bboxes, img_bboxes = get_bboxes(page)
        img_array = np.zeros((doc_h, doc_w), dtype=np.uint8)
        spans_array = np.zeros((doc_h, doc_w), dtype=np.uint8)
        for bbox in span_bboxes:
            bbox = fitz.IRect(bbox) * page.rotation_matrix
            spans_array[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
        spans_array, spans_blob_rects = find_areas(spans_array)
        span_bbox_list.append(spans_blob_rects)
        # save_arr_as_image(spans_array * 255, str(pnum) + '_spans.png')  # << debugging line
        for bbox in img_bboxes:
            bbox = fitz.IRect(bbox) * page.rotation_matrix
            img_array[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
        img_array, img_blob_rects = find_areas(img_array)
        img_bbox_list.append(img_blob_rects)
        # save_arr_as_image(img_array * 255, str(pnum) + '_imgs.png')  # << debugging line

        all_bboxes = np.array([img_array, spans_array, drawings_array]).max(axis=0)
        all_bboxes_list.append(all_bboxes)
        # save_arr_as_image(page_dis * 255, str(pnum) + '_dis.png')  # << debugging line
        all_pages.append(page)
    # <<<< Getting whole document bboxes and finding the Mean >>>>
    doc_mean = np.array(all_bboxes_list).sum(axis=0) / doc.page_count * 255
    # save_arr_as_image(doc_mean * 255, '!!!doc_mean.png')  # << debugging line

    # <<<< Getting whole document bboxes and finding the Mean >>>>
    image = Image.fromarray(doc_mean.astype('uint8'))
    contrast = ImageEnhance.Contrast(image).enhance(1.1)
    # save_arr_as_image(np.asarray(contrast) * 255, '!!!doc_mean_contrast.png')  # << debugging line
    doc_mean_array, doc_mean_blob_rects = find_areas(np.asarray(contrast))
    doc_mean_array, doc_mean_rects = union_rects_throught_array(expand_rects(doc_mean_blob_rects, 2, 2), doc[0].rect.round())
    # save_arr_as_image(doc_mean_array * 255, '!!!doc_mean_array.png')  # <<< debugging line

    # <<<< Sort rects by Area to find the biggest one >>>>
    areas, sorted_blobs = zip(*sorted([(x.get_area(), x) for x in doc_mean_rects], key=lambda x: x[0], reverse=True))
    main_area = sorted_blobs[0]
    exclude_areas = list(sorted_blobs[1:])
    doc_text = []
    for pnum, page in enumerate(doc, 0):
        text_page = []
        p_dict = page.get_text('dict')
        shape = page.new_shape()
        # for area in sorted_blobs:
        #     shape.draw_rect(area * page.derotation_matrix)
        #     shape.finish(color=(0, 0, 0), width=1, fill=(0, 0, 0), fill_opacity=.82)
        #     shape.commit()
        text_blocks = page.get_text('blocks')
        all_exclude_areas = []
        for area in draw_bbox_list[pnum] + img_bbox_list[pnum]:
            inside = False
            for ex_area in exclude_areas:

                if area * page.derotation_matrix in expand_rect(ex_area, 2, 2):
                    inside = True
                    break
            if not inside:
                all_exclude_areas.append(area * page.derotation_matrix)
        for tblock in text_blocks:
            t_rect = fitz.Rect(tblock[:4])
            if t_rect in expand_rect(main_area * page.derotation_matrix, 2, 2):
                exclude = False
                for exclude_rect in all_exclude_areas:
                    if t_rect.intersects(expand_rect(exclude_rect, 5, 5)):
                        exclude = True

                        break
                if not exclude:  # <<<< Here is the area to extract text >>>>
                    # shape.draw_rect(t_rect)
                    # shape.finish(color=(0, 1, 1), width=1, fill=(0, 1, 1), fill_opacity=.2)
                    # shape.commit()
                    for block in p_dict['blocks']:
                        if block['type'] == 0:  # < it's TEXT block
                            rand_block = (random(), random(), random())
                            for line in block['lines']:  # get all spans bboxes
                                if line['bbox'] in t_rect and fitz.sRGB_to_rgb(line['spans'][0]['color']) != (255,255,255):
                                    rand_line = (random(), random(), random())
                                    shape.draw_rect(line['bbox'])
                                    shape.finish(color=rand_line, width=1, fill=rand_block, fill_opacity=.2)
                                    shape.commit()
                                    text_line = []
                                    for snum, span in enumerate(line['spans'], 1):
                                        text_bbox = span['bbox']
                                        if snum == 1:
                                            left_margin = text_bbox[0] - main_area[0]
                                        else:
                                            left_margin = text_bbox[0] - text_line[-1][1][2]
                                        text_line.append((left_margin, text_bbox, span['text'], span['size']))
                                    text_page.append(text_line)

        # <<<< Debugging section "Draw over PDF" >>>>
        #         else:
        #             shape.draw_rect(expand_rect(t_rect, 1, 1))
        #             shape.finish(color=(1, 1, 0), width=1, fill=(1, 1, 0), fill_opacity=.82)
        #             shape.commit()
        #     else:
        #         shape.draw_rect(expand_rect(t_rect, 1, 1))
        #         shape.finish(color=(1, 1, 0), width=1, fill=(1, 1, 0), fill_opacity=.82)
        #         shape.commit()
        # for area in all_exclude_areas:
        #     shape.draw_rect(expand_rect(area, 1, 1))
        #     shape.finish(color=(1, 1, 0), width=1, fill=(1, 1, 0), fill_opacity=.82)
        #     shape.commit()
        # for area in exclude_areas:
        #     shape.draw_rect(area * page.derotation_matrix)
        #     shape.finish(color=(1, 0, 0), width=1, fill=(1, 0, 0), fill_opacity=.82)
        #     shape.commit()
        # shape.draw_rect(main_area * page.derotation_matrix)
        # shape.finish(color=(0, 1, 0), width=1, fill=(1, 0, 0), fill_opacity=0)
        # shape.commit()

        s = 0.5
        for ln, tline in enumerate(text_page, 1):
            if ln == 1:
                prev_y_max = main_area[1]
                sn = '\n'*2
            else:
                prev_y_max = text_page[ln-2][0][1][3]
                sn = ''
            y_min = tline[0][1][1]
            line_size = tline[0][3]
            mt = math.floor((y_min - prev_y_max) / line_size)
            result_text = sn + '\n' * mt
            for span in tline:
                ml, bbox, text, size = span
                space_size = line_size * s
                result_text += ' ' * math.floor(ml / space_size) + text + '\n'
            doc_text.append(result_text)
    # with open(folder_name + '/' + pdf_name + '.txt', 'w', encoding='utf-8') as f:
    #     f.write(''.join(doc_text))

    return {pdf_name : ''.join(doc_text)}
    # print(''.join(doc_text))
    # doc.save(pdf_name + '_result.pdf', garbage=0, clean=True)


def get_bboxes(page):
    span_bboxes = []
    img_bboxes = []
    blocks = page.get_text('dict')['blocks']
    for bnum, block in enumerate(blocks):
        if block['type'] == 0:  # < it's TEXT block
            for line in block['lines']:  # get all spans bboxes
                for span in line['spans']:
                    span_bboxes.append(span["bbox"])
        else:
            img_bboxes.append(block["bbox"])
    return span_bboxes, img_bboxes


def union_rects_throught_array(blocklist, area):
    array = np.zeros((area.height, area.width), dtype=int)
    for rect in blocklist:
        try:
            rect = rect.round()
        except:
            rect = rect
        array[rect[1]:rect[3], rect[0]:rect[2]] = 1
    image, num_areas = ndimage.label(array)
    blobs = ndimage.find_objects(image)
    blob_rects = []
    blob_array = np.zeros((area.height, area.width), dtype=int)
    for blob in blobs:
        y0 = blob[0].start
        y1 = blob[0].stop
        x0 = blob[1].start
        x1 = blob[1].stop
        rect = fitz.IRect(x0, y0, x1, y1)
        blob_rects.append(rect)
        blob_array[blob] = 1
    return blob_array,  blob_rects


def clip(value):
    return 0 if value < 0 else value


def expand_rect(rect, x, y):
    r = rect
    r = fitz.Rect(clip(r[0] - x), clip(r[1] - y), r[2] + x, r[3] + y)
    # x0, y0, x1, y1 = rect
    # r = fitz.Rect(math.ceil(clip(r[0] - x0)), math.ceil(clip(r[1] - y0)), math.floor(x1), math.floor(y1))
    return r


def expand_rects(rect_list, expx, expy):
    temp_list = []
    for rect in rect_list:
        exp_rect = expand_rect(rect, expx, expy)
        temp_list.append(exp_rect)
    return temp_list


def find_areas(array):
    areas_array = np.zeros(array.shape, dtype=np.uint8)
    image, num_areas = ndimage.label(array)
    blobs = ndimage.find_objects(image)
    blob_rects = []
    for blob in blobs:
        y0 = blob[0].start
        y1 = blob[0].stop
        x0 = blob[1].start
        x1 = blob[1].stop
        rect = fitz.IRect(x0, y0, x1, y1)
        blob_rects.append(rect)
        areas_array[blob] = 1
    return areas_array, blob_rects


@app.get("/")
def hello():
    return {'status':'ok'}

@app.post("/process/")
async def pdf_2_text( model:ModelInterface=Depends(),batch: UploadFile=File(...)):
    pdf_name = model.pdf_name
    byte_array = await batch.read()
    folder_name = f'static/{pdf_name[:-4]}_' + datetime.now().strftime(format='%Y-%m-%dT%H.%M.%S')
    os.makedirs(folder_name, exist_ok=True)
    byte_array_doc = byte_array
    with open(f'{folder_name}/{pdf_name}','wb') as f:
        f.write(byte_array_doc)

    rtext = pdf_get_text(pdf_name, folder_name)
    shutil.rmtree(folder_name)


    return {"pdf": rtext}