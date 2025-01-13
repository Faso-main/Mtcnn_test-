import cv2
from mtcnn import MTCNN
 
 
# Создаем экземпляр детектора
detector = MTCNN(device="CPU:0")
 
# Захватываем видео с веб-камеры
cap = cv2.VideoCapture(0)

while True:
    # Читаем кадры из видео
    ret, frame = cap.read()
    if not ret: break
    
    # Применяем детектор к текущему кадру
    result = detector.detect_faces(frame)
 
    # Отображаем результат на кадре
    for face in result:
        x, y, width, height = face['box']
        cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)  # Рисуем прямоугольник
        # Можно также добавить метки для доверия и ключевых точек
        # Доверие
        cv2.putText(frame, f"{face['confidence']:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
 
    # Показываем кадр с результатами
    cv2.imshow('Face Detection', frame)
 
    # Выход при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'): break
 
# Освобождаем объект VideoCapture и закрываем окна
cap.release()
cv2.destroyAllWindows()