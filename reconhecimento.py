
import cv2
import time

# ==============================
# CONFIGURAÇÕES
# ==============================

CAMERA_INDEX = 1

# Cadastro de pessoas voluntárias por QR Code.
# O conteúdo do QR Code deve corresponder a uma destas chaves.
PESSOAS = {
    "ALUNO001": "Pessoa 1",
    "ALUNO002": "Pessoa 2",
    "ALUNO003": "Pessoa 3",
}

TEMPO_EXIBICAO = 5

# ==============================
# INICIAR CÂMERA
# ==============================

camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not camera.isOpened():
    camera.release()
    camera = cv2.VideoCapture(CAMERA_INDEX)

if not camera.isOpened():
    print("Erro: não foi possível abrir a câmera.")
    print("Verifique a conexão USB ou tente CAMERA_INDEX = 1.")
    raise SystemExit

# ==============================
# DETECTOR DE ROSTOS
# ==============================

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

if face_detector.empty():
    print("Erro ao carregar o detector de rostos.")
    camera.release()
    raise SystemExit

# ==============================
# LEITOR DE QR CODE
# ==============================

qr_detector = cv2.QRCodeDetector()

nome_atual = "Aguardando QR Code"
ultimo_tempo = 0

print("Sistema iniciado!")
print("Mostre um QR Code cadastrado para a câmera.")
print("Pressione Q para sair.")

# ==============================
# LOOP PRINCIPAL
# ==============================

while True:
    sucesso, frame = camera.read()

    if not sucesso:
        print("Erro ao capturar imagem da câmera.")
        break

    # Detectar rostos
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rostos = face_detector.detectMultiScale(
        cinza,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(40, 40)
    )

    for (x, y, w, h) in rostos:
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Rosto detectado",
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # Ler QR Code
    dados, pontos, _ = qr_detector.detectAndDecode(frame)

    if dados:
        if dados in PESSOAS:
            nome_atual = PESSOAS[dados]
            ultimo_tempo = time.time()
        else:
            nome_atual = "Codigo nao cadastrado"
            ultimo_tempo = time.time()

    # Voltar à mensagem inicial após alguns segundos
    if time.time() - ultimo_tempo > TEMPO_EXIBICAO:
        nome_atual = "Aguardando QR Code"

    # Mostrar nome ou estado atual
    cv2.putText(
        frame,
        nome_atual,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 200, 0),
        2
    )

    # Mostrar imagem
    cv2.imshow(
        "Reconhecimento em tempo real",
        frame
    )

    # Sair ao pressionar Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==============================
# ENCERRAR
# ==============================

camera.release()
cv2.destroyAllWindows()

print("Programa encerrado.")
