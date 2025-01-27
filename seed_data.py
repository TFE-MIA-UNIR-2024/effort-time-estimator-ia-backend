from supabase_client import get_supabase_client
import datetime

def seed_data():
    supabase = get_supabase_client()

    # Datos a insertar en TipoRequerimiento
    tiporequerimiento_data = [
        {"descripcion": "Tipo de Requerimiento 1"},
        {"descripcion": "Tipo de Requerimiento 2"}
    ]

    try:
        print("Insertando datos en TipoRequerimiento:", tiporequerimiento_data)
        response = supabase.table("tiporequerimiento").insert(tiporequerimiento_data).execute()
        print("Respuesta de la inserción en TipoRequerimiento:", response)
    except Exception as e:
        print("Error al insertar datos en TipoRequerimiento:", e)
        return

    # Datos a insertar en Necesidad
    necesidad_data = [
        {
            "codigonecesidad": "NEC001",
            "nombrenecesidad": "Necesidad 1",
            "proyectoid": 1,
            "fechacreacion": datetime.datetime.now().isoformat()
        },
         {
            "codigonecesidad": "NEC002",
            "nombrenecesidad": "Necesidad 2",
            "proyectoid": 1,
            "fechacreacion": datetime.datetime.now().isoformat()
        }
    ]

    try:
        print("Insertando datos en Necesidad:", necesidad_data)
        response = supabase.table("necesidad").insert(necesidad_data).execute()
        print("Respuesta de la inserción en Necesidad:", response)
    except Exception as e:
        print("Error al insertar datos en Necesidad:", e)
        return
    
    #Datos a Insertar en Tipo Elemento Afectado
    tipo_elemento_afectado_data = [
        {   
            "nombre":"Tipo Elemento Afectado 1",
            "activo": True,
            "fase_proyectoid": 1
        },
        {
            "nombre":"Tipo Elemento Afectado 2",
            "activo": True,
             "fase_proyectoid": 1
        }
    ]
    try:
      print("Insertando datos en Tipo Elemento Afectado:", tipo_elemento_afectado_data)
      response = supabase.table("tipo_elemento_afectado").insert(tipo_elemento_afectado_data).execute()
      print("Respuesta de la inserción en Tipo Elemento Afectado:", response)
      tipo_elemento_afectado_response = response.data
    except Exception as e:
      print("Error al insertar datos en Tipo Elemento Afectado:",e)
      return
    

    # Datos a insertar en Elemento_Afectado
    elemento_afectado_data = [
        {
            "nombre": "Elemento Afectado 1",
            "factor": 1.0,
            "parametro_estimacionid": 1,
            "tipo_elemento_afectadoid":tipo_elemento_afectado_response[0]["tipo_elemento_afectadoid"]
        },
        {
           "nombre": "Elemento Afectado 2",
            "factor": 1.2,
            "parametro_estimacionid": 2,
            "tipo_elemento_afectadoid":tipo_elemento_afectado_response[1]["tipo_elemento_afectadoid"]
        }
    ]
    
    try:
        print("Insertando datos en Elemento_Afectado:", elemento_afectado_data)
        response = supabase.table("elemento_afectado").insert(elemento_afectado_data).execute()
        print("Respuesta de la inserción en Elemento_Afectado:", response)
        elementos_afectados_response = response.data
    except Exception as e:
        print("Error al insertar datos en Elemento_Afectado:", e)
        return
    
    # Datos a insertar en Estimacion Esfuerzo Testing
    estimacion_esfuerzo_testing_data = [
        {"esfuerzorealtesting": 10, "esfuerzoestimadototal": 12},
        {"esfuerzorealtesting": 13, "esfuerzoestimadototal": 14}
    ]
    
    try:
        print("Insertando datos en Estimacion Esfuerzo Testing:", estimacion_esfuerzo_testing_data)
        response = supabase.table("estimacion_esfuerzo_testing").insert(estimacion_esfuerzo_testing_data).execute()
        print("Respuesta de la inserción en Estimacion Esfuerzo Testing:", response)
        estimacion_esfuerzo_testing_response = response.data
    except Exception as e:
        print("Error al insertar datos en Estimacion Esfuerzo Testing:", e)
        return
        
    # Datos a insertar en Estimacion Proyecto
    estimacion_proyecto_data = [
      {
        "codigoestimacion": "EST001",
        "faseestimacion": "Fase 1",
        "fecharequerimiento": datetime.datetime.now().isoformat()
      },
       {
        "codigoestimacion": "EST002",
        "faseestimacion": "Fase 2",
        "fecharequerimiento": datetime.datetime.now().isoformat()
      }
    ]

    try:
        print("Insertando datos en Estimacion Proyecto:", estimacion_proyecto_data)
        response = supabase.table("estimacion_proyecto").insert(estimacion_proyecto_data).execute()
        print("Respuesta de la inserción en Estimacion Proyecto:", response)
        estimacion_proyecto_response = response.data
    except Exception as e:
        print("Error al insertar datos en Estimacion Proyecto:", e)
        return

    # Datos a insertar en Requerimiento
    requerimiento_data = [
        {
            "codigorequerimiento": "REQ001",
            "nombrerequerimiento": "Requerimiento 1",
            "necesidadid": 1,
            "tiporequerimientoid": 1,
            "fechacreacion": datetime.datetime.now().isoformat()
        },
        {
            "codigorequerimiento": "REQ002",
            "nombrerequerimiento": "Requerimiento 2",
            "necesidadid": 2,
             "tiporequerimientoid": 2,
            "fechacreacion": datetime.datetime.now().isoformat()
        }
    ]
    
    try:
        print("Insertando datos en Requerimiento:", requerimiento_data)
        response = supabase.table("requerimiento").insert(requerimiento_data).execute()
        print("Respuesta de la inserción en Requerimiento:", response)
        requerimientos_response = response.data # Obtener el id autogenerado
    except Exception as e:
        print("Error al insertar datos en Requerimiento:", e)
        return

    # Datos a insertar en Punto_Funcion
    punto_funcion_data = [
        {
            "jornada_real": 8,
            "jornada_estimada": 7,
            "cantidad_real": 10,
            "cantidad_estimada": 9,
            "requerimientoid": requerimientos_response[0]["requerimientoid"], # Se usa el id autogenerado
            "parametro_estimacionid": 1,
            "elemento_afectadoid": elementos_afectados_response[0]["elemento_afectadoid"],  # Se usa el id autogenerado
            "estimacion_esfuerzo_testingid": estimacion_esfuerzo_testing_response[0]["estimacion_esfuerzo_testingid"],
            "estimacion_proyectoid": estimacion_proyecto_response[0]["estimacion_proyectoid"]
        },
        {
            "jornada_real": 6,
            "jornada_estimada": 5,
            "cantidad_real": 8,
            "cantidad_estimada": 7,
            "requerimientoid": requerimientos_response[1]["requerimientoid"], # Se usa el id autogenerado
            "parametro_estimacionid": 2,
            "elemento_afectadoid": elementos_afectados_response[1]["elemento_afectadoid"],  # Se usa el id autogenerado
             "estimacion_esfuerzo_testingid": estimacion_esfuerzo_testing_response[1]["estimacion_esfuerzo_testingid"],
             "estimacion_proyectoid": estimacion_proyecto_response[1]["estimacion_proyectoid"]
        }
    ]

    try:
        print("Insertando datos en Punto_Funcion:", punto_funcion_data)
        response = supabase.table("punto_funcion").insert(punto_funcion_data).execute()
        print("Respuesta de la inserción en Punto_Funcion:", response)
        punto_funcion_response = response.data # Obtenemos los ids de la respuesta
    except Exception as e:
        print("Error al insertar datos en Punto_Funcion:", e)
        return
    
   
    # Datos a insertar en Parametro_Estimacion
    parametro_estimacion_data = [
        {
            "nombre": "Factor IA inicial",
            "descripcion": "Factor automático de la IA",
            "factor": 1.0,
            "factor_ia": 1.0,
            "fecha_de_creacion": datetime.datetime.now().isoformat(),
            "pesofactor": 1.0
        },
        {
            "nombre": "Factor IA secundario",
            "descripcion": "Otro factor",
            "factor": 1.2,
            "factor_ia": 1.2,
            "fecha_de_creacion": datetime.datetime.now().isoformat(),
            "pesofactor": 1.1
        }
    ]

    try:
        print("Insertando datos en Parametro_Estimacion:", parametro_estimacion_data)
        response = supabase.table("parametro_estimacion").insert(parametro_estimacion_data).execute()
        print("Respuesta de la inserción en Parametro_Estimacion:", response)
    except Exception as e:
        print("Error al insertar datos en Parametro_Estimacion:", e)
        return
   
    # Datos a insertar en Estimacion_Esfuerzo_Construccion
    estimacion_esfuerzo_construccion_data = [
        {
            "objeto_afectado": "FormularioX",
            "cantidad_objeto_estimado": 4,
            "cantidad_objeto_real": 5,
            "esfuerzo_adicional": 1,
            "justificacion_esfuerzoadicional": "Ajuste de diseño",
            "esfuerzo_real": 12,
            "fechacreacion": datetime.datetime.now().isoformat(),
            "punto_funcionid": punto_funcion_response[0]["punto_funcionid"],  # Se usa el id autogenerado
            "proyectoid": 1
        },
        {
            "objeto_afectado": "ReporteZ",
            "cantidad_objeto_estimado": 2,
            "cantidad_objeto_real": 2,
            "esfuerzo_adicional": 0,
            "justificacion_esfuerzoadicional": "N/A",
            "esfuerzo_real": 8,
            "fechacreacion": datetime.datetime.now().isoformat(),
            "punto_funcionid": punto_funcion_response[1]["punto_funcionid"], # Se usa el id autogenerado
            "proyectoid": 1
        },
        {
            "objeto_afectado": "API_A",
            "cantidad_objeto_estimado": 6,
            "cantidad_objeto_real": 7,
            "esfuerzo_adicional": 2,
            "justificacion_esfuerzoadicional": "Cambios de última hora",
            "esfuerzo_real": 15,
            "fechacreacion": datetime.datetime.now().isoformat(),
            "punto_funcionid": punto_funcion_response[0]["punto_funcionid"], # Se usa el id autogenerado
            "proyectoid": 1
        },
        {
            "objeto_afectado": "ServicioB",
            "cantidad_objeto_estimado": 5,
            "cantidad_objeto_real": 5,
            "esfuerzo_adicional": 1,
            "justificacion_esfuerzoadicional": "Refactor crítico",
            "esfuerzo_real": 10,
            "fechacreacion": datetime.datetime.now().isoformat(),
            "punto_funcionid": punto_funcion_response[0]["punto_funcionid"], # Se usa el id autogenerado
            "proyectoid": 1
        },
        {
            "objeto_afectado": "MóduloIntegraciónC",
            "cantidad_objeto_estimado": 3,
            "cantidad_objeto_real": 4,
            "esfuerzo_adicional": 1,
            "justificacion_esfuerzoadicional": "Ajustes de compatibilidad",
            "esfuerzo_real": 11,
            "fechacreacion": datetime.datetime.now().isoformat(),
            "punto_funcionid": punto_funcion_response[0]["punto_funcionid"], # Se usa el id autogenerado
            "proyectoid": 1
        },
        {
            "objeto_afectado": "MigraciónD",
            "cantidad_objeto_estimado": 8,
            "cantidad_objeto_real": 9,
            "esfuerzo_adicional": 3,
            "justificacion_esfuerzoadicional": "Nuevos requerimientos",
            "esfuerzo_real": 20,
            "fechacreacion": datetime.datetime.now().isoformat(),
            "punto_funcionid": punto_funcion_response[0]["punto_funcionid"],  # Se usa el id autogenerado
            "proyectoid": 1
        }
    ]

    try:
        print("Insertando datos en Estimacion_Esfuerzo_Construccion:", estimacion_esfuerzo_construccion_data)
        response = supabase.table("estimacion_esfuerzo_construccion").insert(estimacion_esfuerzo_construccion_data).execute()
        print("Respuesta de la inserción en Estimacion_Esfuerzo_Construccion:", response)
    except Exception as e:
        print("Error al insertar datos en Estimacion_Esfuerzo_Construccion:", e)
        return

if __name__ == "__main__":
    seed_data()