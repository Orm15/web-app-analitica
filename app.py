from flask import Flask, render_template, request, session, redirect, url_for, jsonify
import psycopg2
import json
from unidecode import unidecode
import re
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import numpy as np
import nltk
import fitz
import os
import google.generativeai as genai


app = Flask(__name__)
app.secret_key = 'tu_clave_secreta_segura'  # Configura la clave secreta al inicio

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')  # Carpeta 'uploads' en el directorio actual
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Crear la carpeta si no existe
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Verificar si los stopwords ya están descargados
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

sw = set(stopwords.words('spanish'))
host = "postgresql://DBPF-Analitica_owner:p6dlCyb1PoTV@ep-tight-frog-a5tijn15.us-east-2.aws.neon.tech/DBPF-Analitica?sslmode=require"

@app.route('/')
def log():
    return render_template('menuInicio.html')  

@app.route('/main1')
def index1():  # Empresa
    return render_template('PrincipalEmpresa.html')

@app.route('/main2')
def index2():  # Postulante
    return render_template('PrincipalPostulante.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Lógica del logeo
        if username == "string" and password == "string":
            session['username'] = username  # Guarda el nombre de usuario en la sesión
            return redirect(url_for('index1'))
        else:
            error = 'Nombre de usuario o contraseña incorrectos. Intenta de nuevo.'
            return render_template('login.html', error=error)

    return render_template('login.html')

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page_num in range(pdf.page_count):
            page = pdf[page_num]
            text += page.get_text()
    return text

@app.route('/register')
def GoRegister():
    return render_template('register.html')

@app.route('/UserRegister', methods=['POST'])
def UserRegister():
    # Lógica de registro
    return redirect(url_for('login'))

def limpiarOracionBiDireccional(oracion):
    minuscula = oracion.lower()
    SinStopWords = " ".join([palabra for palabra in minuscula.split() if palabra not in sw])
    especiales = unidecode(SinStopWords)
    caracteresE = " ".join(re.findall(r'\b[a-zA-Z0-9]+\b',especiales))
    return caracteresE

def textoAvector(texto, modelo):
    palabras = texto.split()
    vectores = [modelo.wv[palabra] for palabra in palabras if palabra in modelo.wv]
    return np.mean(vectores, axis=0) if vectores else np.zeros(modelo.vector_size)

def getPerfilesCercanos(vector_entrada):
    vector_entrada_str = ",".join(map(str, vector_entrada))
    # Conectar a la base de datos
    try:
        conn = psycopg2.connect(host)
        cursor = conn.cursor()
        print("Conexión exitosa a Neon")

        query = f"""
        SELECT id, nombre, link, educacion, experiencia, habilidades,
            vector_representativo <-> cube(ARRAY[{vector_entrada_str}]) AS distancia
        FROM perfiles
        ORDER BY distancia
        LIMIT 10;
        """
        # Ejecutar la consulta
        cursor.execute(query)
        # Obtener los resultados
        resultados = cursor.fetchall()

        cursor.close()
        conn.close()
        return resultados

    except Exception as e:
        print("Error al conectar a Neon o al obtener los datos:", e)
    return ["Algo salió mal"]

## Funcion para traer n empleados
@app.route('/generar_tabla', methods=['POST'])
def generar_tabla():

    # Cargo el modelo
    modelo = Word2Vec.load("models/modelo_w2vPerfiles.model")
    texto1 = request.form['texto1']
    texto2 = request.form['texto2']
    texto3 = request.form['texto3']
    
    # Convertir campos a vector
    vector_requisitos = (
        (np.mean([textoAvector(limpiarOracionBiDireccional(texto1), modelo) ,
        textoAvector(limpiarOracionBiDireccional(texto2), modelo) ,
        textoAvector(limpiarOracionBiDireccional(texto3), modelo)], axis = 0)).tolist()
    )

    salida = getPerfilesCercanos(vector_requisitos)

    # Prepara los datos para ser enviados como JSON
    datos = [{'nombre': row[1], 'link': row[2], 'educación': row[3],
              'experiencia': row[4], 'habilidades': row[5]} for row in salida]
    
    return jsonify(datos)
########################################################################################################
os.environ["API_KEY"] = 'AIzaSyCpORLezb5oUrutWv4bXkBzOiWjOCdkoGc'
genai.configure(api_key=os.environ["API_KEY"])

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-pro-002",
  generation_config=generation_config,
)

chat_session = model.start_chat(
  history=[
  ]
)

def GetCaracteristicas(texto):
    limpieza = texto.lower()
    match = re.search(
        r"\*\*educación requerida:?\*\*(.*?)\*\*experiencia requerida:?\*\*(.*?)\*\*habilidades o conocimientos:?\*\*(.*?)\*\*tipo de puesto:?\*\*(.*)",
        limpieza, re.DOTALL
    )
    
    if match:
        edu, exp, hab, tip = match.groups()
        return edu.strip(), exp.strip(), hab.strip(), tip.strip()
    else:
        return None, None, None, None

def getCategorias(texto):
    prompt = "extrae lo siguiente: la educacion requerida, experiencia requerida, las habilidades o conocimientos y tipo de puesto(presencial, virtual o hibrido) en ese orden, usa el siguiente texto:  "
    prompt += texto + ". Respeta el formato "
    response = chat_session.send_message(prompt)
    r = (response.candidates[0].content.parts[0].text)
    return GetCaracteristicas(r)

def getPuestosCercanos(vector_entrada, n = 10):
    # Convertir el vector de entrada a formato de cadena para pasarlo en la consulta
    vector_entrada_str = ",".join(map(str, vector_entrada))

    # Conectar a la base de datos
    try:
        conn = psycopg2.connect(host)
        cursor = conn.cursor()
        print("Conexión exitosa a Neon")

        query = f"""
        SELECT id, puesto, linkpuesto, educacion, experiencia, habilidades, tipo, categoria,
            vectorCaracteristico <-> cube(ARRAY[{vector_entrada_str}]) AS distancia
        FROM puestos
        ORDER BY distancia
        LIMIT {n};
        """

        # Ejecutar la consulta
        cursor.execute(query)

        # Obtener los resultados
        resultados = cursor.fetchall()

        cursor.close()
        conn.close()
        return resultados
    except Exception as e:
        print("Error al conectar a Neon o al obtener los datos:", e)
    return []

## Funcion para traer n Puestos
@app.route('/generar_tabla2', methods=['POST'])
def generar_tabla2():
    if request.is_json:
        data = request.get_json()
        cantidad = data.get('texto2')
        nombreArchivo = data.get('nombre')
    else:
        cantidad = request.form.get('texto2')  # Para datos enviados como form-data
        nombreArchivo = request.form['nombre']

    # Cargo el modelo
    modelo = Word2Vec.load("models/modelo_w2v.model")
   
    
    ## Extraigo el texto:
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], nombreArchivo)
    texto_extraido = extract_text_from_pdf(file_path)

    ## Uso gemini para separar:

    datosPostulante = getCategorias(texto_extraido)
    limpio = [limpiarOracionBiDireccional(lineas) for lineas in datosPostulante[:3]]

    vecs = [textoAvector(campo, modelo) for campo in limpio]
    resumen = sum(vecs)

    val = getPuestosCercanos(resumen,cantidad)

    # Prepara los datos para ser enviados como JSON
    datos = [{'puesto': row[1], 'link': row[2], 'experiencia': row[3],
              'habilidades': row[4], 'educacion': row[5]} for row in val]
    
    return jsonify(datos)

# Ruta para subir el archivo
@app.route('/subir-pdf', methods=['POST'])
def subir_pdf():
    if 'pdf_file' not in request.files:
        return 'No file part', 400
    
    pdf_file = request.files['pdf_file']
    
    if pdf_file.filename == '':
        return redirect(url_for('index2'))

    # Guardar el archivo PDF en la carpeta de uploads
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_file.filename)
    pdf_file.save(file_path)

    return '', 204

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    # Ejecuta en 0.0.0.0 para que Render pueda acceder
    app.run(host="0.0.0.0", port=port)
