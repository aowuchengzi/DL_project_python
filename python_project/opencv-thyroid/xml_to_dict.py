# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 19:35:07 2018

@author: 叶晨
"""
#%%
from lxml import etree
import numpy as np
#%%
def recursive_parse_xml_to_dict(xml):
  """Recursively parses XML contents to python dict.
  We assume that `object` tags are the only ones that can appear
  multiple times at the same level of a tree.
  Args:
    xml: xml tree obtained by parsing XML file contents using lxml.etree 通过使用lxml.etree解析XML文件内容而获得的xml树
  Returns:
    Python dictionary holding XML contents.
  """
  if not len(xml):
    return {xml.tag: xml.text}
  result = {}
  for child in xml:
    child_result = recursive_parse_xml_to_dict(child)
    if child.tag != 'object':
      result[child.tag] = child_result[child.tag]
    else:
      if child.tag not in result:
        result[child.tag] = []
      result[child.tag].append(child_result[child.tag])
  return {xml.tag: result}
#%%
def xmlfile_to_dict(path):
    with open(path, 'r') as flie:
        xml_text = flie.read()
        xml = etree.fromstring(xml_text)
        xml_dict = recursive_parse_xml_to_dict(xml)
    return xml_dict
#%%
def get_bndbox_from_dict(xml_dict):
    width = int(xml_dict['annotation']['size']['width'])
    height = int(xml_dict['annotation']['size']['height'])
    label = [] #[xmin, ymin, xmax, ymax]
    label.append(float(xml_dict['annotation']['object'][0]['bndbox']['xmin']) / width)
    label.append(float(xml_dict['annotation']['object'][0]['bndbox']['ymin']) / height)
    label.append(float(xml_dict['annotation']['object'][0]['bndbox']['xmax']) / width)
    label.append(float(xml_dict['annotation']['object'][0]['bndbox']['ymax']) / height)
    label_array = np.array(label)
    return label_array
#%%
#with open(r"F:\picture\localization\image_output\Annotations\b1.xml", 'r') as file:
#    print(file.read())
#    xml_text = file.read()
#    xml = etree.fromstring(xml_text)
#    aa = recursive_parse_xml_to_dict(xml)
##%%
#dir(aa)
#aa['annotation']['object'][0]['bndbox']['xmax']
##%%
#width = int(aa['annotation']['size']['width'])
#height = int(aa['annotation']['size']['height'])
#
#
#label = []
#label.append(float(aa['annotation']['object'][0]['bndbox']['xmin']) / width)
#label.append(float(aa['annotation']['object'][0]['bndbox']['ymin']) / height)
#label.append(float(aa['annotation']['object'][0]['bndbox']['xmax']) / width)
#label.append(float(aa['annotation']['object'][0]['bndbox']['ymax']) / height)
#
#a = []
#a.append(label)
#
##%%
#xml = etree.parse(r"F:\picture\localization\image_output\Annotations\b1.xml") #读取xml文件
#root = xml.getroot()
#print(root.items())
#print(root.keys())
#print(root.get('verified'))
##%%
#for node in root.getchildren():
#    print(node.tag) #输出节点的标签名
#
#dir(root)
#bndbox = xml.xpath('//bndbox')
#for node in bndbox.getchildren():
#    print(node.tag) #输出节点的标签名
#type(bndbox)
#type(root)
#%%






