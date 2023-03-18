import PySimpleGUI as sg
import os.path
from PIL import Image
from io import BytesIO
import torch
import yaml
import shutil
import cv2
from PIL import ExifTags
import seaborn


#Sekcja głównego interfejsu

#Wybór folderu ze zdjęciami do analizy
folder_list_column = [ #kolumna odpowiedzialna za wybór folderu
    [
        sg.Text("Folder ze zdjęciami do analizy"),
        sg.In(size=(25,1), enable_events=True, key="-IMAGE FOLDER-"),
        sg.FolderBrowse()
    ],
    [
        sg.Listbox(values=[], enable_events=True, size=(40,20), key="-IMAGE LIST-")
    ]
]

si_list_column = [ #kolumna odpowiedzialna za wybór customowego modelu
    [
        sg.Text("Wybór modelu SI"),
        sg.In(size=(30,1), enable_events=True, key="-SI-"),
        sg.FileBrowse()
    ]
]

output_list_column = [ #kolumna odpowiedzialna za wybór folderu zapisu zdjęć po analizie
    [
        sg.Text("Folder wyjściowy"),
        sg.In(size=(25,1), enable_events=True, key="-OUTPUT FOLDER-"),
        sg.FolderBrowse(),
    ],
    [
        sg.Button("Wykonaj analizę", key="-WORK-")
    ]
]

image_view_column = [ #podgląd zdjęcia 
    [sg.Text("Wybierz zdjęcie z listy po lewej: ")],
    [sg.Text(size=(60,2), key="-IMAGE VIEW PATH-")],
    [sg.Image(key="-IMAGE VIEW-")]
]

# basic_layout = [ #model stwrzony z powyższych kresek
#     [
#         sg.Column(folder_list_column),
#         sg.VSeparator(), #gustowna kreska oddzielająca sekcje
#         sg.Column(si_list_column),
#         sg.VSeparator(),
#         sg.Column(output_list_column),
#         sg.VSeparator(),
#         sg.Column(image_view_column)
#     ]
# ]

#Koniec sekcji głównego interfejsu

#Sekcja interfejsu uczenia modelu oraz zaznaczania klas na zdjęciu

image_folder_training = [ #folder ze zdjęciami do zaznaczania klas
    [
        sg.Text("Folder ze zdjęciami do treningu"),
        sg.In(size=(25,1), enable_events=True, key="-IMAGE FOLDER MODEL-"),
        sg.FolderBrowse()
    ],
    [
        sg.Listbox(values=[], enable_events=True, size=(40,20), key="-FILE LIST MODEL-")
    ]
]

class_list = [ #kolumna odpowiedzialna za pokazywanie dostępnych klas zdefiniowanych w pliku dataset.yaml oraz za button do rozpoczęcia treningu (trzeba dodać możliwość wyboru ilości powtórzeń itd.)
    [
        sg.Listbox(values=[], enable_events=True, size=(40,20), key="-CLASS LIST-")
    ],
    [
        sg.Text('Ilość generacji: ', size=(15, 1)), sg.Spin(values=[i for i in range(1, 5000)], initial_value=5, size=(6, 1), key="-GENERATIONS-"),
        sg.Button("Rozpocznij Trening", key="-TRAINING-")
    ],
    [
        sg.Button("Załaduj klasy", key="-TRAIN-") #a może by to przenieść do kolumny SI?
    ]
]

# modelTrainingLayout = [ #interfejs uczenia modelu (dodać powrót do poprzedniego menu)
#     [
#         sg.Column(image_folder_training),
#         sg.VSeparator(),
#         sg.Graph( #okno umożliwiające zaznaczanie na zdjęciu
#             canvas_size=(400,400),
#             graph_bottom_left=(0,0),
#             graph_top_right=(400,400),
#             key="-GRAPH-",
#             change_submits=True, #rejestracja wduszania guziora myszy
#             background_color='lightblue',
#             drag_submits=True, #rejestracja przeciagania myszą
#         ),
#         sg.VSeparator(),
#         sg.Column(class_list)
#     ],
#     [
#         sg.Text(key="-INFO-", size=(60,1)) #informacja gdzie zostało zaznaczone (do wyrzucenia?)
#     ]
# ]

#Koniec sekcji interfejsu uczenia modelu

#Połączony interfejs

ui = [ 
    [
        sg.Column(folder_list_column, scrollable=True),
        sg.VSeparator(), #gustowna kreska oddzielająca sekcje
        sg.Column(si_list_column),
        sg.VSeparator(),
        sg.Column(output_list_column),
        sg.VSeparator(),
        sg.Column(image_view_column)
    ],
    [sg.HorizontalSeparator()],
    [
        sg.Column(image_folder_training),
        sg.VSeparator(),
        sg.Graph( #okno umożliwiające zaznaczanie na zdjęciu
            canvas_size=(400,400),
            graph_bottom_left=(0,0),
            graph_top_right=(400,400),
            key="-GRAPH-",
            change_submits=True, #rejestracja wduszania guziora myszy
            background_color='lightblue',
            drag_submits=True, #rejestracja przeciagania myszą
        ),
        sg.VSeparator(),
        sg.Column(class_list)
    ],
    [sg.HorizontalSeparator()],
    [
        sg.Text(key="-INFO-", size=(60,1)) #informacja gdzie zostało zaznaczone (do wyrzucenia?)
    ]
]


#Koniec

                                                                                                                        #tu zmień rozmiar
window =sg.Window("Analiza zdjęć", resizable=True, auto_size_buttons=True, auto_size_text=True).Layout([[sg.Column(ui, size=(1920,1080) ,scrollable=False)]]).Finalize() #ustawienie okna pod stworzony intrefejs (na starcie podstawowy, potem po naciśnięciu odpowiedniego guziora zmieniany)
window.bind('<Configure>', "-EVENT-") #nie wiem co to robi, ale jest potrzebne żeby można było wyłapywać event przy zwiększeniu okna

#zmienne Globalne
windowSize = window.size #wielkość okna

filename = "" #ścieżka do zdjęcia/folderu
isFolder = False #bool sprawdzający czy powyższa ścieżka jest folderem
output = "" #folder w którym będzie zapisywany
customSi = "" #ścieżka do wybranego modelu, jeśli puste analiza wykona się na predefiniowanym modelu 
file_list = "" #lista plików w folderu
classFile = "" #plik zawierający klasy w nowym modelu
classList = {} #lista klas w nowym modelu
classID = 0 #wybrana na liście przez użytkownika klasa, potrzebna do zapisu w pliku

dragging = False #czy jest przeciągnay kursor na grafie
start_point = None #punkt początkowy zaznaczenia na grafie
end_point = None #punkt końcowy zaznaczenia na grafie
prior_rect = None #wielkość zaznaczenia

trainPath = "dataset/" #ścieżka z plikami do treningu

#funkcje odpowiedzialne za działanie poniższego programu

def convertToPng(image): #konwersja zdjęcia do png
    with BytesIO() as f:
        image.save(f, format='PNG')
        return f.getvalue()

def openImage(image):
    i = Image.open(image)
    width, height = i.size
    scaleFactor = 400/height #skalujemy zdjęcie żeby interfejs nie rozjechał się jak oczy zezowatego
    i = i.resize((int(width*scaleFactor), 400), Image.ANTIALIAS) #wysokość zawsze 400px, szerokośc zależna od skali liczonej wyżej 
    i = convertToPng(i) #potrzeba, aby otworzyć zdjęcie, pysimplegui ma problem z jpg
    return i

def setFilesFromFolder(folder):
    try:
        return os.listdir(folder) #próbujemy dostać wszystkie pliki dostępne w tym folderze
    except:
        return [] #w przeciwnym wypadku dajemy pustą listę. Z tego co pamiętam, bez tego try/excepta to wywalało błąd, więc musi bo program się dusi


def setFolderContent(folder, file_list, windowName): #zmienna folder jest ustawiana na folder wybrany przez użytkownika przez przycisk
    fnames = [ #lista plików zawierających odpowiednie rozszerzenie (w sumie można by było ją rozwinąć, ale to szczegół)
        f
        for f in file_list
        if os.path.isfile(os.path.join(folder, f))
        and f.lower().endswith((".png", ".gif", ".jpg", ".jpeg"))
    ]
    window[windowName].update(fnames) #pokazanie listy znalezionych plików spełaniających warunki w linijkach wyżej - czytaj zdjęć
    if len(fnames): #jeśli jest więcej niż 0 zdjęć
        if windowName == "-IMAGE LIST-": #dla głównego interfejsu, pokaż zdjęcie
            showImageInPreview(folder, fnames[0]) #pokaż pierwsze zdjęcie z folderu

def showImageInPreview(folder, fileName): #pokazujemy zdjęcie które
    file = os.path.join(folder, fileName) #łączenie ścieżki folderu
    print(file)
    window["-IMAGE VIEW PATH-"].update("Obecny podgląd pliku to {0}".format(fileName.replace("\\", "/"))) #pokaż ścieżke pliku którego widzimy podgląd (przy łączeniu używając os.path.join z jakiegoś powodu dodaje się \ a nie / jak w reszcie ścieżki jest, stąd zmiana replace)
    window["-IMAGE VIEW-"].update(openImage(file)) #wyświetlamy zdjecie. Funkcja openImage konwertuje na png, żeby pysimplegui mogłosobie poradzić, bez tego to chyba tylko png i gify otwiera, a jpg nie 

def usePredefiniedModel(f): #użyj predefiniowanego modelu - yolov5x
    model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True) #torch pobiera podany model biblioteki yolo (pewnie można robić to z pliku, ale po co?)
    results = model(f)
    results.save(output)

def pickFilesToAnalize(filename, file_list): #wybieranie
    for files in file_list: #dla każdego pliku z list
        path = os.path.join(filename, files) #połącz z nazwą folderu
        usePredefiniedModel(path) #użyj predefiniowanego modelu


def getClassesFromFile(classFile): #funkcja odpowiedzialna za otworzenie interfejsu treningu
    classes = yaml.safe_load(classFile) #załaduj dane z pliku 
    classes = classes["names"] #zapisz w zmiennej tylko nazwy klass
    window["-CLASS LIST-"].update(values = classes) #wyświetl w liście nazwy klas
    classList = {} #pusty słownik, potem używany do zapisywanie id klasy w pliku do treningu
    for i in range(len(classes)):
        classList[classes[i]] = i #nazwa jako klucz, id jako wartosć
    return classList

def openScaledImage(filename): #otwórz zdjecie 400x400 zawsze, tak aby pasowało do grafu
    im = Image.open(filename)
    im = im.resize((400, 400), Image.ANTIALIAS) 
    image = convertToPng(im)
    return image

def drawImageOnGraph(f, graph):
    graph.draw_image(data=openScaledImage(f), location=(0,400)) if f else None

def getOriginalSizeImage(filename): #zwraca oryginalną wielkość zdjęcia
    im = Image.open(filename)
    width, height = im.size
    im.close()
    return (width, height)

def calculatePoints(point, scale): #przeskalowuje punkty na oryginalną wielkość zdjęcia
    x = (0,0)
    try:
        x = (point[0] * scale[0], point[1] * scale[1])
    except:
        x = (0,0)
    return x

def getRectangleData(sp, ep): #zapis w postaci tablicy rzeczy potrzebnych dla treningu. odpowiednio, ID klasy, środek klasy na osi X, środek klasy na osi Y, wielkość na osi X, wielkośc na osi Y
    xCenter = (sp[0]+ep[0])/2
    yCenter = (sp[1]+ep[1])/2
    xWidth = abs(sp[0]-ep[0])
    yHeight = abs(sp[1]-ep[1])
    printBuffer = ("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(classID, xCenter, yCenter, xWidth, yHeight))
    printBuffer = printBuffer.split(" ")
    return printBuffer

def refractorData(table): #zapisywanie woli z jakiejś przyczyny zapis pojedynczego stringa w tabeli
    string = "{} {} {} {} {}".format(table[0], table[1], table[2], table[3], table[4])
    return [string]

def normalize(point, size):
    return (point[0]/size[0], point[1]/size[1])

#i koniec funkcji


#crème de la crème - sam program
while True:
    event, values = window.read() #sprawdzanie który event się zdarzył, oraz wartosci dla eventów
    if event == "Exit" or event == sg.WIN_CLOSED: #miał być tu match/case (chyba taki switch w pythonie) ale on jest od 3.10 XD a to pisane jest w 3.9 😠
        break
    if event == "-IMAGE FOLDER-": #jeśli event ma wartość taki jak klucz to to się wykonuje. Mam nadzieję że reszty ifów nie muszę tłumaczyć XD
        filename = values["-IMAGE FOLDER-"] #nazwa folderu wybranego przez użytkownika
        isFolder = True 
        file_list = setFilesFromFolder(filename)
        setFolderContent(filename, file_list, "-IMAGE LIST-")
    if event == "-IMAGE LIST-": #jeśli została naciśnięta lista ze zdjęciami
        try: #łapanie wyjątku, naciśnięcia na pustą liste, bez wybranego folderu
            filename = os.path.join(values["-IMAGE FOLDER-"], values["-IMAGE LIST-"][0])
            isFolder = False
            showImageInPreview("", filename)
        except:
            None #no bo po co ma walić errorami? niech nic nie robi Nop();
    if event == "-OUTPUT FOLDER-":
        output = values["-OUTPUT FOLDER-"] #zapis wybranego folderu do zapisu
    if event == "-WORK-":
        
        if not os.path.exists("yolov5"):
            cmd = "git clone https://github.com/ultralytics/yolov5"
            os.system(cmd)
        if output != "" and filename != "": #musi być przypisany folder output
            if customSi == "": #jeśli nie został wybrany inny model
                if isFolder: #jeśli jest to folder
                    pickFilesToAnalize(filename, file_list)
                else:
                    if filename != "": #jeśli nazwa pliku istnieje, tak dla pewności
                        usePredefiniedModel(filename)
                        sg.popup_ok('Program zakończył analizę')
            elif customSi.endswith(".pt"):
                if not os.path.exists("dataset"):
                    os.mkdir("dataset")
                if not os.path.exists("dataset/images"):
                    os.mkdir("dataset/images")
                if not os.path.exists("dataset/images/train"):
                    os.mkdir("dataset/images/train")
                if not os.path.exists("dataset/images/val"):
                    os.mkdir("dataset/images/val")
                if not os.path.exists("dataset/labels"):
                    os.mkdir("dataset/labels")
                if not os.path.exists("dataset/labels/train"):
                    os.mkdir("dataset/labels/train")
                if not os.path.exists("dataset/labels/val"):
                    os.mkdir("dataset/labels/val")
                if not os.path.exists("dataset/dataset.yaml"):
                    with open("dataset/dataset.yaml", "a") as f:
                        print("path: ../dataset\ntrain: images/train\nval: images/train\ntest:\n\nnc: 2\nnames: ['jellyfish', 'bird']", file=f)


                runCustomModelCommand = "python yolov5/detect.py --source " + filename + " --weights " + customSi + " --data dataset/dataset.yaml --project "+ output +" --name detectTest" #komenda pozwalająca na użycie customowego modelu, wymaga pythona
                os.system(runCustomModelCommand) #wykonanie tej komendy
                if filename != "":
                    sg.popup_ok('Program zakończył analizę')
            else:
                sg.popup_ok('Wybrano nieprawidłowy plik jako model!')
        else:
            sg.popup_ok('Najpierw musisz wybrać folder zapisu oraz folder z plikami!')
    if event == "-TRAIN-":
        #window.close() #zamknij bieżące okno (główne, gdzie można wykonać analizę)
        #window = sg.Window("Trening modelu", modelTrainingLayout, resizable=True, auto_size_buttons=True, auto_size_text=True).Finalize()
        #window.bind('<Configure>', "-EVENT MODEL-") #do zmiany wielkości okan
        #windowSize = window.size


        if not os.path.exists("dataset"):
            os.mkdir("dataset")
        if not os.path.exists("dataset/images"):
            os.mkdir("dataset/images")
        if not os.path.exists("dataset/images/train"):
            os.mkdir("dataset/images/train")
        if not os.path.exists("dataset/images/val"):
            os.mkdir("dataset/images/val")
        if not os.path.exists("dataset/labels"):
            os.mkdir("dataset/labels")
        if not os.path.exists("dataset/labels/train"):
            os.mkdir("dataset/labels/train")
        if not os.path.exists("dataset/labels/val"):
            os.mkdir("dataset/labels/val")
        if not os.path.exists("dataset/dataset.yaml"):
            with open("dataset/dataset.yaml", "a") as f:
                print("path: ../dataset\ntrain: images/train\nval: images/train\ntest:\n\nnc: 2\nnames: ['jellyfish', 'bird']", file=f)


        classFile = open("dataset/dataset.yaml") #otwórz plik z definicjami klas, do zmiany na dynamiczne później
        classList = getClassesFromFile(classFile) #zmień okno interfejsu
        classFile.close()
    if event == "-IMAGE FOLDER MODEL-":
        filename = values["-IMAGE FOLDER MODEL-"] #nazwa folderu wybranego przez użytkownika
        isFolder = True 
        file_list = setFilesFromFolder(filename)
        setFolderContent(filename, file_list, "-FILE LIST MODEL-")
    if event == "-FILE LIST MODEL-":
        try: #łapanie wyjątku, naciśnięcia na pustą liste, bez wybranego folderu
            filename = os.path.join(values["-IMAGE FOLDER MODEL-"], values["-FILE LIST MODEL-"][0])
            isFolder = False
            graph = window["-GRAPH-"] #weź element o kluczu -GRAPH-
            drawImageOnGraph(filename, graph)
        except:
            None 
    if event == "-CLASS LIST-":
        try:
            choosenClass = values["-CLASS LIST-"][0] #values zawsze zwracane są jako lista
            classID = classList[choosenClass] #znalezienie ID klasy po nazwie (kluczu)
        except:
            None
    if event == "-GRAPH-":
        x,y = values["-GRAPH-"] #wartości gdzie było naciśnięte na grafie
        if not dragging:
            start_point = (x,y) #punkt początkowy ustaw na x,y
            dragging = True
        else:
            end_point = (x,y) #punkt końcowy ustaw na x,y
        if prior_rect: #jeśli zaznaczenie istnieje
            graph.delete_figure(prior_rect) #usuń figurę z grafu
        if None not in (start_point, end_point): #jeśli mamy oba punkty ustawione
            try:
                prior_rect = graph.draw_rectangle(start_point, end_point,line_color='red') #rysuj prostokąt, od punktu startu do końca, czerwoy
            except:
                None
    if event.endswith("+UP"): #jeśli event konczy się puszczeniem przycisku
        try:
            info = window["-INFO-"] #okno z tekstem informacyjnym o zaznaczeniu


            dragging = False
            image_Size = getOriginalSizeImage(filename) #wielkośc potrzebna do skalowania odpowiedniego punktów
            drawingScale = (image_Size[0]/400, image_Size[1]/400) #400 ponieważ taką wielkość miał graf
            
            sp = calculatePoints(start_point, drawingScale) #punkt na oryginalnej wielkości
            ep = calculatePoints(end_point, drawingScale) #drugi punkt na oryginalnej wielkości

            info.update(value=f"Prostokąt od {sp} do {ep}") #informacje nt. punktu na oryginalnej wielkości zdjęcia
            
            sp = normalize(sp, image_Size) #normalizacja punktów
            ep = normalize(ep, image_Size)
            
            newName = filename.split("\\")[-1] #nowa nazwa dla pliku
            newName = newName.split(".")[0] + ".txt" #usuwamy rozszerzenie dodajemy .txt
            
            saveClassPath = os.path.join(trainPath, "labels/train/") #ścieżka do plików txt z labelsami potrzebna do treningu, musi nazwać się tak samo jak zdjecie (oprócz rozszerzenia)
            save_file_name = os.path.join(saveClassPath, newName) #nazwa pliku, tam gdzie będzie zapisane
            printBuffer = getRectangleData(sp, ep) #dane nt. zaznacznia w postaci tego jak chce w treningu
            printData = refractorData(printBuffer) #dane zapisane tak, aby można było je zapisać do pliku
            if float(printBuffer[3]) < 0.001 and float(printBuffer[4]) < 0.001: #jeśli jedna z długości jest mniejsza, ma ich nie zapisywać (czasem po szybkim kliku wskakują wartości o długosci (0,0), trzeba je odrzucać)
                start_point, end_point = None, None
            else: 
                print('\n'.join(printData), file=open(save_file_name, 'a+')) #zapis do pliku wartości o klasie
                start_point, end_point = None, None #reset wartości punktów
                path1 = os.path.join(filename) #ścieżka do pliku (potrzebna przy kopiowaniu z inputu do images/train/)
                
                f = filename.replace("\\", "/").split("/")
                tp = os.path.join(trainPath, "images/train/") #ścieżka do folderu treningowego
                path2 = os.path.join(tp, str(f[-1])) #gdzie ma skopiować i o jakiej nazwie
                print(path1)
                print(path2)
                shutil.copyfile(path1, path2)
        except:
            None
    if event == "-TRAINING-":
        numOfGeneraton = int(values["-GENERATIONS-"])

        
        if not os.path.exists("yolov5/train.py"):
            cmd = "git clone https://github.com/ultralytics/yolov5"
            os.system(cmd)

        command = "python yolov5/train.py --img 640 --cfg yolov5/models/yolov5s.yaml --hyp yolov5/data/hyps/hyp.scratch-med.yaml --batch 64 --epochs {} --data dataset/dataset.yaml --weights yolov5/yolov5s.pt --workers 24 --name yolo_jellyfish_det --project /VTF --device cpu".format(numOfGeneraton)
        os.system(command) #będzi tu trzeba zmienić pewne szczególy, ustawianie ilości generacji, itd.
        sg.popup_ok('Program zakończył trening')

    if event == "-SI-":
        customSi = str(values["-SI-"]) #ściezka do SI
    if event == "-EVENT-":
        if(window.size == windowSize): #jeśli wielkość nadal taka sama
            continue
        else:
            if(window.size[0] > windowSize[0]): #mniejsza
                window["-IMAGE LIST-"].set_size((40, 80))
            else:
                window["-IMAGE LIST-"].set_size((40,20))
    if event == "-EVENT MODEL-":
        if(window.size == windowSize): #jeśli wielkość nadal taka sama
            continue
        else:
            if(window.size[0] > windowSize[0]): #mniejsza
                window["-FILE LIST MODEL-"].set_size((40, 80))
                window["-CLASS LIST-"].set_size((40, 80))
            else:
                window["-FILE LIST MODEL-"].set_size((40,20))
                window["-CLASS LIST-"].set_size((40, 20))

window.close() #zamknięcie okna