"""Simple parser for the annotation XML files used in this project.

This parser is tailored to files like the example provided by the user:

<annotation>
  <folder>...</folder>
  <filename>...</filename>
  <size>
    <width>...</width>
    <height>...</height>
    <depth>...</depth>
  </size>
  <object>  # optional
    <name>...</name>
    <unique_id>...</unique_id>
    <bndbox>
       <xmin>...</xmin>
       <xmax>...</xmax>
       <ymin>...</ymin>
       <ymax>...</ymax>
    </bndbox>
  </object>
</annotation>

The parser returns a normalized dict with keys: `image`, `size`, `entities`, and `relationships`.
"""
from __future__ import annotations

import os
import json
from typing import Any, Dict, Optional
import xml.etree.ElementTree as ET


def _text(e: Optional[ET.Element]) -> Optional[str]:
    if e is None or e.text is None:
        return None
    return e.text.strip()


def _int(e: Optional[ET.Element]) -> Optional[int]:
    t = _text(e)
    if t is None:
        return None
    try:
        return int(t)
    except Exception:
        return None


def parse_annotation_xml(path: str) -> Dict[str, Any]:
    """Parse the annotation XML at `path` and return a normalized dict.

    - Uses `<filename>` as `image` (falls back to xml basename + .jpg).
    - Extracts `<size>` fields into `size` dict.
    - Extracts `<object>` entries (if present) into `entities` using `<unique_id>` when available.
    - Each entity contains `id`, `type` (from <name>), and `bounds` as [xmin,ymin,xmax,ymax].

    The parser is tolerant to the absence of `<object>`.
    """
    tree = ET.parse(path)
    root = tree.getroot()

    out: Dict[str, Any] = {
        "image": None,
        "size": {},
        "entities": {},
        "relationships": {},
    }

    # image filename
    fn = root.find("filename")
    if fn is not None and _text(fn):
        out["image"] = _text(fn)
    else:
        base = os.path.basename(path)
        stem = os.path.splitext(base)[0]
        out["image"] = f"{stem}.jpg"

    # size
    size_node = root.find("size")
    if size_node is not None:
        width = _int(size_node.find("width"))
        height = _int(size_node.find("height"))
        depth = _int(size_node.find("depth"))
        if width is not None:
            out["size"]["width"] = width
        if height is not None:
            out["size"]["height"] = height
        if depth is not None:
            out["size"]["depth"] = depth

    # objects (optional)
    # There may be zero or many <object> nodes; handle all.
    for obj in root.findall("object"):
        name = _text(obj.find("name")) or "object"
        unique_id = _text(obj.find("unique_id"))

        # if unique_id not provided, fall back to box_id or generated key
        eid = unique_id or _text(obj.find("box_id")) or f"obj_{len(out['entities'])}"

        # bounds
        b = obj.find("bndbox")
        bounds = None
        if b is not None:
            xmin = _int(b.find("xmin")) or 0
            ymin = _int(b.find("ymin")) or 0
            xmax = _int(b.find("xmax")) or xmin
            ymax = _int(b.find("ymax")) or ymin
            bounds = [xmin, ymin, xmax, ymax]

        out["entities"][eid] = {
            "id": eid,
            "type": name,
            "label": name,
            "bounds": bounds,
        }

    return out


if __name__ == "__main__":
    import argparse
    from tqdm import tqdm
    from shutil import copyfile
    import pandas as pd
    # p = argparse.ArgumentParser()
    # p.add_argument("xml",required=False, help="Path to annotation xml")
    # args = p.parse_args()
    lesion_info_file = r'C:\Users\50183\Downloads\lesion_info.csv'
    # Load lesion info as dict
    lesion_df = pd.read_csv(lesion_info_file)
    lesion_info = lesion_df.set_index('unique_object_id')['histology_class'].to_dict()
    # # need to skip the header
    # with open(lesion_info_file, 'r') as f:
    #     lines = f.readlines()
    #     lines = lines[1:]  # skip header
    #     for line in lines:
    #         parts = line.strip().split(',')
    #         if len(parts) >= 2:
    #             lesion_info[parts[0]] = parts[-1]  # lesion_id: histology_class
    print(f"Loaded lesion info for {len(lesion_info)} lesions.")

    annotation_xml_dir = "C:/Users/50183/Downloads/001-001_annotations/001-001_annotations"
    images_dir = r'C:\Users\50183\Downloads\001-001_frames\001-001_frames'
    output_dir = r'C:\COMP9800\Dataset\real-colon-seg'
    xml_files = [f for f in os.listdir(annotation_xml_dir) if f.endswith(".xml")]
    img_with_lesions = []
    for xml_file in tqdm(xml_files):
        file_path = os.path.join(annotation_xml_dir, xml_file)
        parsed = parse_annotation_xml(file_path)
        if parsed['entities']:
            img_with_lesions.append(parsed['image'])
            img_path = os.path.join(images_dir, parsed['image'])
            target_img_path = os.path.join(output_dir,'images', parsed['image'])
            target_annotation_path = os.path.join(output_dir,'annotations', xml_file.replace('.xml','.json'))
            if os.path.exists(img_path):
                # Copy the image to output_dir
                copyfile(img_path, target_img_path)
                copyfile(file_path, os.path.join(output_dir,'annotations', xml_file))
                json.dump(parsed, open(target_annotation_path,'w'), indent=4)
                # save the parsed bbox as csv file: class_name, xmin, ymin, xmax, ymax
                csv_file_path = os.path.join(output_dir, 'bbox', xml_file.replace('.xml','.csv'))
                with open(csv_file_path, 'w') as f:
                    f.write("class_name,xmin,ymin,xmax,ymax\n")
                    for lesion_id, entity in parsed['entities'].items():
                        if entity['bounds']:
                            class_name = lesion_info.get(lesion_id, 'polyp')
                            xmin, ymin, xmax, ymax = entity['bounds']
                            f.write(f"{class_name},{xmin},{ymin},{xmax},{ymax}\n")
    print('Annotated images:',len(img_with_lesions))