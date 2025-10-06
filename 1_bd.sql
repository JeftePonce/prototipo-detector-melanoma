-- Tabla de Pacientes
CREATE TABLE pacientes (
    id_paciente SERIAL PRIMARY KEY,
    rut VARCHAR(12) UNIQUE NOT NULL,
    nombre VARCHAR(100) NOT NULL,
    apellido VARCHAR(100) NOT NULL,
    fecha_nacimiento DATE NOT NULL,
    edad INT,
    telefono VARCHAR(15),
    email VARCHAR(100),
    direccion TEXT,
    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    activo BOOLEAN DEFAULT TRUE
);

-- Tabla de Consultas
CREATE TABLE consultas (
    id_consulta SERIAL PRIMARY KEY,
    id_paciente INT NOT NULL,
    fecha_consulta DATE NOT NULL,
    medico_tratante VARCHAR(100),
    notas_consulta TEXT,
    localizacion_lesion VARCHAR(200),
    tipo_lesion VARCHAR(100),
    tamano_lesion VARCHAR(50),
    color_lesion VARCHAR(100),
    diagnostico_clinico VARCHAR(200),
    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (id_paciente) REFERENCES pacientes(id_paciente) ON DELETE CASCADE
);

-- Tabla de Imágenes
CREATE TABLE imagenes (
    id_imagen SERIAL PRIMARY KEY,
    id_consulta INT NOT NULL,
    nombre_archivo VARCHAR(255) NOT NULL,
    ruta_almacenamiento VARCHAR(500) NOT NULL,
    fecha_captura TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    tipo_imagen VARCHAR(10), -- 'jpg', 'png', etc.
    tamaño_archivo BIGINT, -- en bytes
    resolucion VARCHAR(20), -- '1920x1080'
    calidad_imagen VARCHAR(50),
    FOREIGN KEY (id_consulta) REFERENCES consultas(id_consulta) ON DELETE CASCADE
);

-- Tabla de Resultados del Modelo
CREATE TABLE resultados_modelo (
    id_resultado SERIAL PRIMARY KEY,
    id_imagen INT NOT NULL,
    probabilidad_melanoma DECIMAL(5,4), -- 0.0000 a 1.0000
    diagnostico_modelo VARCHAR(50), -- 'Melanoma', 'No melanoma', etc.
    confianza DECIMAL(5,4),
    caracteristicas_analizadas JSONB, -- JSONB es más eficiente en PostgreSQL
    fecha_analisis TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    version_modelo VARCHAR(50),
    FOREIGN KEY (id_imagen) REFERENCES imagenes(id_imagen) ON DELETE CASCADE
);