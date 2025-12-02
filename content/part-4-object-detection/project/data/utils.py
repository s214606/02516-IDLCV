from xml.etree import ElementTree as ET

def read_content(xml_file: str):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []

    for boxes in root.iter('object'):

        filename = root.find('filename').text

        ymin, xmin, ymax, xmax = None, None, None, None
        class_id = (boxes.find("name").text)
        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)

        # list_with_single_boxes = [xmin, ymin, xmax, ymax]
        # list_with_all_boxes.append(list_with_single_boxes)
        
        label_map = {class_id: 1}   # later you can do {"dog": 1, "cat": 2, ...}
        class_id = label_map[class_id]   # -> 1
        print(f"la clase de esta madre es {class_id}")
        list_with_single_boxes = [xmin, ymin, xmax, ymax, class_id]
        list_with_all_boxes.append(list_with_single_boxes)

    return filename, list_with_all_boxes