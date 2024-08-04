import numpy as np

def group_text_box(polys, slope_ths=0.25, ycenter_ths=0.4, height_ths=0.6, width_ths=0.5, add_margin=0.05, sort_output=True):
    horizontal_list, free_list, combined_list, merged_list = [], [], [], []

    for poly in polys:
        # Lấy các điểm từ polygon
        x1, y1 = poly[0]
        x2, y2 = poly[1]
        x3, y3 = poly[2]
        x4, y4 = poly[3]

        # Tính độ dốc của các cạnh trên và dưới
        slope_up = (y2 - y1) / np.maximum(10, (x2 - x1))
        slope_down = (y3 - y4) / np.maximum(10, (x3 - x4))
        
        if max(abs(slope_up), abs(slope_down)) < slope_ths:
            # Nếu polygon gần như nằm ngang, lấy bounding box của nó
            x_max = max(x1, x2, x3, x4)
            x_min = min(x1, x2, x3, x4)
            y_max = max(y1, y2, y3, y4)
            y_min = min(y1, y2, y3, y4)
            horizontal_list.append([x_min, x_max, y_min, y_max, 0.5 * (y_min + y_max), y_max - y_min])
        else:
            # Với các polygon không nằm ngang, thêm chúng vào danh sách free_list
            height = np.linalg.norm([x4 - x1, y4 - y1])
            width = np.linalg.norm([x2 - x1, y2 - y1])

            margin = int(1.44 * add_margin * min(width, height))

            theta13 = abs(np.arctan((y1 - y3) / np.maximum(10, (x1 - x3))))
            theta24 = abs(np.arctan((y2 - y4) / np.maximum(10, (x2 - x4))))

            x1_adj = x1 - np.cos(theta13) * margin
            y1_adj = y1 - np.sin(theta13) * margin
            x2_adj = x2 + np.cos(theta24) * margin
            y2_adj = y2 - np.sin(theta24) * margin
            x3_adj = x3 + np.cos(theta13) * margin
            y3_adj = y3 + np.sin(theta13) * margin
            x4_adj = x4 - np.cos(theta24) * margin
            y4_adj = y4 + np.sin(theta24) * margin

            free_list.append([[x1_adj, y1_adj], [x2_adj, y2_adj], [x3_adj, y3_adj], [x4_adj, y4_adj]])

    if sort_output:
        horizontal_list = sorted(horizontal_list, key=lambda item: item[4])

    # Kết hợp và gộp các box nằm ngang
    new_box = []
    for poly in horizontal_list:
        if len(new_box) == 0:
            b_height = [poly[5]]
            b_ycenter = [poly[4]]
            new_box.append(poly)
        else:
            if abs(np.mean(b_ycenter) - poly[4]) < ycenter_ths * np.mean(b_height):
                b_height.append(poly[5])
                b_ycenter.append(poly[4])
                new_box.append(poly)
            else:
                b_height = [poly[5]]
                b_ycenter = [poly[4]]
                combined_list.append(new_box)
                new_box = [poly]
    combined_list.append(new_box)

    # Gộp các box trong cùng một dòng
    for boxes in combined_list:
        if len(boxes) == 1:  # Single box per line
            box = boxes[0]
            margin = int(add_margin * min(box[1] - box[0], box[5]))
            merged_list.append([[box[0] - margin, box[2] - margin], [box[1] + margin, box[2] - margin], [box[1] + margin, box[3] + margin], [box[0] - margin, box[3] + margin]])
        else:  # Multiple boxes per line
            boxes = sorted(boxes, key=lambda item: item[0])

            merged_box, new_box = [], []
            for box in boxes:
                if len(new_box) == 0:
                    b_height = [box[5]]
                    x_max = box[1]
                    new_box.append(box)
                else:
                    if (abs(np.mean(b_height) - box[5]) < height_ths * np.mean(b_height)) and \
                            ((box[0] - x_max) < width_ths * (box[3] - box[2])):  # Merge boxes
                        b_height.append(box[5])
                        x_max = box[1]
                        new_box.append(box)
                    else:
                        b_height = [box[5]]
                        x_max = box[1]
                        merged_box.append(new_box)
                        new_box = [box]
            if len(new_box) > 0:
                merged_box.append(new_box)

            for mbox in merged_box:
                if len(mbox) != 1:  # Adjacent boxes in the same line
                    x_min = min(mbox, key=lambda x: x[0])[0]
                    x_max = max(mbox, key=lambda x: x[1])[1]
                    y_min = min(mbox, key=lambda x: x[2])[2]
                    y_max = max(mbox, key=lambda x: x[3])[3]

                    box_width = x_max - x_min
                    box_height = y_max - y_min
                    margin = int(add_margin * (min(box_width, box_height)))

                    merged_list.append([[x_min - margin, y_min - margin], [x_max + margin, y_min - margin], [x_max + margin, y_max + margin], [x_min - margin, y_max + margin]])
                else:  # Non-adjacent box in the same line
                    box = mbox[0]

                    box_width = box[1] - box[0]
                    box_height = box[3] - box[2]
                    margin = int(add_margin * (min(box_width, box_height)))

                    merged_list.append([[box[0] - margin, box[2] - margin], [box[1] + margin, box[2] - margin], [box[1] + margin, box[3] + margin], [box[0] - margin, box[3] + margin]])

    all_boxes = merged_list + free_list
    return all_boxes
