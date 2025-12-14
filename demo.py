
import tempfile
import cv2
import torch
import streamlit as st
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, './yolov5')
import streamlit as st
import time
import IPython
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_boxes, 
                                  check_imshow, xyxy2xywh, increment_path)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import os
import streamlit as st
import torch
import argparse
  

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# Global variables for counting objects
data_car = []
data_bus = []
data_truck = []
data_motor = []
already = []
line_pos = 0.6

def detect(opt, stframe, car, bus, truck, motor, line, fps_rate, class_id):
    out, source, yolo_model, deep_sort_model, show_vid, save_vid, save_txt, imgsz, evaluate, half, project, name, exist_ok= \
        opt.output, opt.source, opt.yolo_model, opt.deep_sort_model, opt.show_vid, opt.save_vid, \
        opt.save_txt, opt.imgsz, opt.evaluate, opt.half, opt.project, opt.name, opt.exist_ok
    # choose custom class from streamlit
    opt.classes = class_id
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')
    sum_fps = 0
    line_pos = line
    save_vid = True
    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(deep_sort_model,
                        max_dist=cfg.DEEPSORT.MAX_DIST,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn)
    stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(save_dir)) + '/' + txt_file_name + '.txt'

    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
        t1 = time_sync()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t2 = time_sync()
        dt[0] += t2 - t1

        prev_time = time.time()
        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if opt.visualize else False
        pred = model(img, augment=opt.augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)
        dt[2] += time_sync() - t3

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            s += '%gx%g ' % img.shape[2:]  # print string

            annotator = Annotator(im0, line_width=2, pil=not ascii)
            w, h = im0.shape[1],im0.shape[0]
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]
                
                # pass detections to deepsort
                t4 = time_sync()
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4

                # draw boxes for visualization
                if len(outputs) > 0:
                    for j, (output, conf) in enumerate(zip(outputs, confs)):

                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]
                        #count
                        c = int(cls)  # integer class
                        label = f'{id} {names[c]} {conf:.2f}'
                        annotator.box_label(bboxes, label, color=colors(c, True))
                        # count_obj(bboxes,w,h,id, names[c], data_car, data_bus, data_truck, data_motor)
                        count_obj(bboxes,w,h,id, names[c], line_pos)
                        
                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path, 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))

                LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')

            else:
                deepsort.increment_ages()
                LOGGER.info('No detections')

            # Stream results
            im0 = annotator.result()
            if show_vid:
                # count vehicle
                color = (0,255,0)
                color_car = (0,150,255)
                color_bus = (0,255,0)
                color_truck = (255,0,0)
                color_motor = (255,255,0)
                start_point = (0, int(line_pos*h))
                end_point = (w, int(line_pos*h))
                cv2.line(im0, start_point, end_point, color, thickness=2)
                thickness = 3
                org = (20, 70)
                distance_height = 100
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 2
                # cv2.putText(im0, 'car: ' + str(len(data_car)), org, font, fontScale, color_car, thickness, cv2.LINE_AA)
                # cv2.putText(im0, 'bus: ' + str(len(data_bus)), (org[0], org[1] + distance_height), font, fontScale, color_bus, thickness, cv2.LINE_AA)
                # cv2.putText(im0, 'truck: ' + str(len(data_truck)), (org[0], org[1] + distance_height*2), font, fontScale, color_truck, thickness, cv2.LINE_AA)
                # cv2.putText(im0, 'motor: ' + str(len(data_motor)), (org[0], org[1] + distance_height*3), font, fontScale, color_motor, thickness, cv2.LINE_AA)

                # cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_vid:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 60, im0.shape[1], im0.shape[0]

                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

                vid_writer.write(im0)

                # show fps
                curr_time = time.time()
                fps_ = curr_time - prev_time
                fps_ = round(1/round(fps_, 3),1)
                prev_time = curr_time
                sum_fps += fps_

                stframe.image(im0, channels="BGR", use_column_width=True)
                car.markdown(f"<h3> {str(len(data_car))} </h3>", unsafe_allow_html=True)
                bus.write(f"<h3> {str(len(data_bus))} </h3>", unsafe_allow_html=True)
                truck.write(f"<h3> {str(len(data_truck))} </h3>", unsafe_allow_html=True)
                motor.write(f"<h3> {str(len(data_motor))} </h3>", unsafe_allow_html=True)
                fps_rate.markdown(f"<h3> {fps_} </h3>", unsafe_allow_html=True)
    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    print("Average FPS", round(1 / (sum(list(t)) / 1000), 1))
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms deep sort update \
        per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        print('Results saved to %s' % save_path)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)
    
def load_counts_from_file():
    """Loads counts of vehicles from a text file."""
    if os.path.exists('vehicle_count.txt'):
        with open('vehicle_count.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'Car count:' in line:
                    data_car.extend([i for i in range(int(line.split(': ')[1].strip()))])
                elif 'Bus count:' in line:
                    data_bus.extend([i for i in range(int(line.split(': ')[1].strip()))])
                elif 'Truck count:' in line:
                    data_truck.extend([i for i in range(int(line.split(': ')[1].strip()))])
                elif 'Motorcycle count:' in line:
                    data_motor.extend([i for i in range(int(line.split(': ')[1].strip()))])



def save_counts_to_file():
    """Saves the counts of vehicles to a text file."""
    with open('C:\\Users\\jashk\\OneDrive\\Desktop\\Vehicle_Detection_and_Counting_System-main\\vehicle_count.txt', 'w') as f:
        f.write(f"Car count: {len(data_car)}\n")
        f.write(f"Bus count: {len(data_bus)}\n")
        f.write(f"Truck count: {len(data_truck)}\n")
        f.write(f"Motorcycle count: {len(data_motor)}\n")

def reset_counts():
    """Resets the vehicle counts to zero."""
    global data_car, data_bus, data_truck, data_motor
    data_car, data_bus, data_truck, data_motor = [], [], [], []
    save_counts_to_file()  # Save the reset counts to the file

def count_obj(box, w, h, id, label, line_pos):
    global data_car, data_bus, data_truck, data_motor, already
    center_coordinates = (int(box[0] + (box[2] - box[0]) / 2), int(box[1] + (box[3] - box[1]) / 2))

    # Classify one time per id
    if center_coordinates[1] > (h * line_pos):
        if id not in already:
            already.append(id)
            if label == 'car' and id not in data_car:
                data_car.append(id)
            elif label == 'bus' and id not in data_bus:
                data_bus.append(id)
            elif label == 'truck' and id not in data_truck:
                data_truck.append(id)
            elif label == 'motorcycle' and id not in data_motor:
                data_motor.append(id)

            # Save counts to file after counting a new vehicle
            save_counts_to_file()

def main():
    global is_running
    st.title('Vehicle Detection and Counting')
    st.markdown('<h3 style="color: red">with Yolov5 and Deep SORT</h3>', unsafe_allow_html=True)
    load_counts_from_file() 
    # Video upload and setting up
    video_file_buffer = st.sidebar.file_uploader("Upload a video", type=['mp4', 'mov', 'avi'])

    if video_file_buffer:
        st.sidebar.text('Input video')
        st.sidebar.video(video_file_buffer)
        video_path = os.path.join('videos', video_file_buffer.name)
        with open(video_path, 'wb') as f:
            f.write(video_file_buffer.getbuffer())

    # Custom class selection
    custom_class = st.sidebar.checkbox('Custom classes')
    assigned_class_id = [0, 1, 2, 3]
    names = ['car', 'motorcycle', 'truck', 'bus']

    if custom_class:
        assigned_class_id = []
        assigned_class = st.sidebar.multiselect('Select custom classes', names)
        for each in assigned_class:
            assigned_class_id.append(names.index(each))

    confidence = st.sidebar.slider('Confidence', min_value=0.0, max_value=1.0, value=0.5)
    line = st.sidebar.number_input('Line position', min_value=0.0, max_value=1.0, value=0.6, step=0.1)

    status = st.empty()
    stframe = st.empty()

    car, bus, truck, motor = st.columns(4)
    
    with car:
        st.markdown('**Car**')
        car_text = st.markdown('__')
    
    with bus:
        st.markdown('**Bus**')
        bus_text = st.markdown('__')

    with truck:
        st.markdown('**Truck**')
        truck_text = st.markdown('__')
    
    with motor:
        st.markdown('**Motorcycle**')
        motor_text = st.markdown('__')

    fps_col, _, _, _ = st.columns(4)
    
    with fps_col:
        st.markdown('**FPS**')
        fps_text = st.markdown('__')

    # Start, Stop, and Reset buttons
    track_button = st.sidebar.button('START')
    stop_button = st.sidebar.button('STOP')
    reset_button = st.sidebar.button('RESET')

    if track_button:
        is_running = True
       
        opt = parse_opt()
        opt.conf_thres = confidence
        opt.source = video_path

        status.markdown('<font size="4"> **Status:** Running... </font>', unsafe_allow_html=True)
        
        with torch.no_grad():
            while is_running:
                detect(opt, stframe, car_text, bus_text, truck_text, motor_text, line, fps_text, assigned_class_id)
            
            # Ensure the status updates when the loop exits
            status.markdown('<font size="4"> **Status:** Stopped </font>', unsafe_allow_html=True)

    # Handle the Stop button click
    if stop_button:
        is_running = False
        status.markdown('<font size="4"> **Status:** Stopped </font>', unsafe_allow_html=True)

        # Save vehicle counts to a file
        save_counts_to_file()

        # Read saved counts and display in the corresponding columns
        with open('C:\\Users\\jashk\\OneDrive\\Desktop\\Vehicle_Detection_and_Counting_System-main\\vehicle_count.txt', 'r') as f:
            lines = f.readlines()
            car_count = int(lines[0].split(":")[1].strip())
            bus_count = int(lines[1].split(":")[1].strip())
            truck_count = int(lines[2].split(":")[1].strip())
            motorcycle_count = int(lines[3].split(":")[1].strip())

            car_text.markdown(f"**{car_count}**")
            bus_text.markdown(f"**{bus_count}**")
            truck_text.markdown(f"**{truck_count}**")
            motor_text.markdown(f"**{motorcycle_count}**")

            # Estimation section
            average_co2_emissions = {
                'Car': 120,       # Average CO2 emissions per car in grams/km
                'Bus': 300,       # Average CO2 emissions per bus in grams/km
                'Truck': 400,     # Average CO2 emissions per truck in grams/km
                'Motorcycle': 80   # Average CO2 emissions per motorcycle in grams/km
            }

            total_emission = (car_count * average_co2_emissions['Car'] +
                            bus_count * average_co2_emissions['Bus'] +
                            truck_count * average_co2_emissions['Truck'] +
                            motorcycle_count * average_co2_emissions['Motorcycle'])

            estimation_text = "### Estimation of Total CO2 Emissions"
            car_emission_text = f"Cars: {car_count} x {average_co2_emissions['Car']} g/km = {car_count * average_co2_emissions['Car']} g/km"
            bus_emission_text = f"Buses: {bus_count} x {average_co2_emissions['Bus']} g/km = {bus_count * average_co2_emissions['Bus']} g/km"
            truck_emission_text = f"Trucks: {truck_count} x {average_co2_emissions['Truck']} g/km = {truck_count * average_co2_emissions['Truck']} g/km"
            motorcycle_emission_text = f"Motorcycles: {motorcycle_count} x {average_co2_emissions['Motorcycle']} g/km = {motorcycle_count * average_co2_emissions['Motorcycle']} g/km"

            # Display the estimation texts
            st.markdown(estimation_text)
            st.markdown(car_emission_text)
            st.markdown(bus_emission_text)
            st.markdown(truck_emission_text)
            st.markdown(motorcycle_emission_text)
            total_population_text = f"**Total CO2 Emission:** {total_emission} g/km"
            st.markdown(total_population_text)


    # Handle the Reset button click
    if reset_button:
        reset_counts()
        st.success("Counts have been reset to zero.")





def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_model', nargs='+', type=str, default='best_new.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_model', type=str, default='osnet_x0_25')
    parser.add_argument('--source', type=str, default='videos/motor.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[480], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_false', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='evaluate inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--project', default='runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # Expand
    return opt

if __name__ == "__main__":
    main()
