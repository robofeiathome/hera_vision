# Yolov8_Tracking e flow+track (Portuguese)

## Aviso 

O arquivo responsável pela funcionalidade de track+flow ainda está em desenvolvimento. No entanto, agora, o flow está funcionando sem erros maiores, mas ainda preciso adicionar algumas coisas para que ele funcione em todo o seu potencial. Algumas melhorias estão a caminho e em breve serão adicionadas a este Git.

## Descrição

Este programa utiliza a engine da YOLOv8 e oferece alguns programas de tracking, sendo definido por padrão como StrongSORT. Para a detecção, o programa utiliza o peso da YOLOv8 para segmentação por padrão, mas também é possível utilizar pesos de detecção normais ou pesos de segmentação personalizados, alterando o caminho e o nome do peso. Este método apresentou uma perda de ID menor em comparação com o DeepSORT, porém ainda não é perfeito e, às vezes, ocorre perda de ID.

## Instalação

__Caso deseje usar uma GPU para rodar o programa, certifique-se de que seu CUDA esteja instalado e configurado corretamente.__ Para verificar se seu CUDA está instalado e qual a sua versão, rode o comando abaixo no terminal:
```
$ nvcc --version
```
Caso não funcione, instale o NVIDIA Toolkit. Tutorial de instalação disponível no [site da Nvidia](https://developer.nvidia.com/cuda-downloads).

### track

Para facilitar o setup e uso do programa, um arquivo "requirements.txt" foi adicionado. Primeiro, vá até o repositório e entre na pasta do Yolov8_Tracking.
```
$ cd hera_vision/
$ cd Yolov8_Tracking/
```
Em seguida, instale os requerimentos.
```
$ pip install -r requirements.txt
```
Pronto, os pacotes para rodar o track estão prontos, mas ainda é necessário instalar as bibliotecas para que o YOLO rode.

### Yolo

Ainda na pasta "Yolov8_Tracking", vá para a pasta do yolov8.
```
$ cd yolov8/
```
E por fim, instale tanto o Ultralytics quanto os requerimentos.
```
$ pip install ultralytics
$ pip install -r requirements.txt
```
Agora, todos os pacotes para rodar o programa estão instalados.

## Uso 

Você pode rodar o programa de tracking de duas maneiras: pelo arquivo Python usando as configurações setadas nele, ou informando os parâmetros na linha de comando.

### Arquivo .py

No terminal, vá até a pasta Yolov8_Tracking e rode o programa.
```
$ python3 track.py
```
Em alguns segundos, o programa começará a rodar. Ele irá identificar o dispositivo CUDA (caso exista) e informará tanto o nome quanto a memória dedicada disponível. Caso não encontre um dispositivo CUDA, ele informará que está rodando com a CPU.

### Alterando os parametros

No terminal, vá até a pasta Yolov8_Tracking. Dentro da pasta, basta rodar como no exemplo anterior do arquivo .py, mas adicionando na frente do comando o parâmetro a ser definido e o valor a ser dado para ele. Por exemplo:
```
$ python3 track.py --yolo-weights yolov8n.pt # bboxes 
                        yolov8n-seg.pt  # bboxes + segmentacao
```
Todos os parâmetros que podem ser alterados estão listados dentro do código entre as linhas 383 e 419. Sempre que for adicionar mais um parâmetro e um valor, seguir o mesmo padrão, dando espaço entre o valor anterior e o próximo parâmetro. Exemplo:
```
$ python3 track.py --source 0 --yolo-weights yolov8n.pt --img 640
```
Também será possível ver os parâmetros aqui no arquivo README.md, que serão listados em seguida:

## Parametros 

__Segue a lista dos argumentos e suas definições padrão. Alguns recebem True ou False, enquanto outros recebem valores numéricos (exceto pelo CUDA, que pode receber "cpu")__

 - yolo-weights (default=WEIGHTS / 'yolov8s-seg.pt')
 - reid-weights (default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
 - tracking-method (default='bytetrack')
 - tracking-config (type=Path, default=None)
 - source (default='0')
 - imgsz (default=[640])
 - conf-thres (default=0.5)
 - iou-thres (default=0.5)
 - max-det (default=1000)
 - device (default='')
 - show-vid (action='store_true')
 - save-txt (action='store_true')
 - save-conf (action='store_true')
 - save-crop (action='store_true')
 - save-trajectories (action='store_true')
 - save-vid (action='store_true')
 - nosave (action='store_true')
 - classes
 - agnostic-nms (action='store_true')
 - augment (action='store_true')
 - visualize (action='store_true')
 - update (action='store_true')
 - project (default=ROOT / 'runs' / 'track')
 - name (default='exp')
 - exist-ok (action='store_true')
 - line-thickness (default=2)
 - hide-labels (default=False, action='store_true')
 - hide-conf (default=False, action='store_true')
 - hide-class (default=False, action='store_true')
 - half (action='store_true')
 - dnn (action='store_true')
 - vid-stride (type=int, default=1)
 - retina-masks (action='store_true')
 - 
# Yolov8_Tracking e flow+track (English)

## Warning

The file responsible for the track+flow functionality is still under development. However, now the flow is working without major errors, but I still need to add some things for it to work at its full potential. Some improvements are on the way and will soon be added to this Git.

## Description

This program uses the YOLOv8 engine and offers some tracking programs, with StrongSORT set as the default. For detection, the program uses YOLOv8's segmentation weights by default, but it's also possible to use normal detection weights or custom segmentation weights by changing the weight path and name. This method showed a lower ID loss compared to DeepSORT, but it's not perfect, and ID loss sometimes occurs.

## Installation

__If you want to use a GPU to run the program, make sure your CUDA is installed and configured correctly.__ To check if your CUDA is installed and its version, run the command below in the terminal:
```
$ nvcc --version
```
If it doesn't work, install the NVIDIA Toolkit. Installation tutorial available on the [Nvidia website](https://developer.nvidia.com/cuda-downloads).

### track

To facilitate the setup and use of the program, a "requirements.txt" file has been added. First, go to the repository and enter the Yolov8_Tracking folder.
```
$ cd hera_vision/
$ cd Yolov8_Tracking/
```
Next, install the requirements.
```
$ pip install -r requirements.txt
```
Now, the packages to run the track are ready, but you still need to install the libraries for YOLO to run.

### Yolo

Still in the "Yolov8_Tracking" folder, go to the yolov8 folder.
```
$ cd yolov8/
```
Finally, install both Ultralytics and the requirements.
```
$ pip install ultralytics
$ pip install -r requirements.txt
```
Now, all the packages to run the program are installed.

## Usage

You can run the tracking program in two ways: by the Python file using the settings set in it, or by providing the parameters in the command line.

### .py File

In the terminal, go to the Yolov8_Tracking folder and run the program.
```
$ python3 track.py
```
In a few seconds, the program will start running. It will identify the CUDA device (if any) and inform both the name and the available dedicated memory. If it doesn't find a CUDA device, it will report that it is running with the CPU.

### Changing Parameters

In the terminal, go to the Yolov8_Tracking folder. Inside the folder, just run as in the previous example of the .py file, but adding in front of the command the parameter to be set and the value to be given to it. For example:
```
$ python3 track.py --yolo-weights yolov8n.pt # bboxes 
                        yolov8n-seg.pt  # bboxes + segmentacao
```
All the parameters that can be changed are listed within the code between lines 383 and 419. Whenever you want to add another parameter and a value, follow the same pattern, leaving space between the previous value and the next parameter. Example:
```
$ python3 track.py --source 0 --yolo-weights yolov8n.pt --img 640
```
You can also see the parameters here in the README.md file, which will be listed below:

## Parameters

__Here is the list of arguments and their default settings. Some accept True or False, while others accept numerical values (except for CUDA, which can take "cpu")__

 - yolo-weights (default=WEIGHTS / 'yolov8s-seg.pt')
 - reid-weights (default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
 - tracking-method (default='bytetrack')
 - tracking-config (type=Path, default=None)
 - source (default='0')
 - imgsz (default=[640])
 - conf-thres (default=0.5)
 - iou-thres (default=0.5)
 - max-det (default=1000)
 - device (default='')
 - show-vid (action='store_true')
 - save-txt (action='store_true')
 - save-conf (action='store_true')
 - save-crop (action='store_true')
 - save-trajectories (action='store_true')
 - save-vid (action='store_true')
 - nosave (action='store_true')
 - classes
 - agnostic-nms (action='store_true')
 - augment (action='store_true')
 - visualize (action='store_true')
 - update (action='store_true')
 - project (default=ROOT / 'runs' / 'track')
 - name (default='exp')
 - exist-ok (action='store_true')
 - line-thickness (default=2)
 - hide-labels (default=False, action='store_true')
 - hide-conf (default=False, action='store_true')
 - hide-class (default=False, action='store_true')
 - half (action='store_true')
 - dnn (action='store_true')
 - vid-stride (type=int, default=1)
 - retina-masks (action='store_true')
 - 
