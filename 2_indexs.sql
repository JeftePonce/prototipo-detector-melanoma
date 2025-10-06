-- Índices para la tabla consultas
CREATE INDEX idx_consultas_fecha ON consultas(fecha_consulta);
CREATE INDEX idx_consultas_paciente_fecha ON consultas(id_paciente, fecha_consulta);

-- Índices para la tabla imagenes
CREATE INDEX idx_imagenes_consulta ON imagenes(id_consulta);
CREATE INDEX idx_imagenes_fecha ON imagenes(fecha_captura);

-- Índices para la tabla resultados_modelo
CREATE INDEX idx_resultados_imagen ON resultados_modelo(id_imagen);
CREATE INDEX idx_resultados_diagnostico ON resultados_modelo(diagnostico_modelo);