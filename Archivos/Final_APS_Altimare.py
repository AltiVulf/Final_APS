# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 15:49:43 2026

@author: Tomás Altimare Bercovich

Descripción:

"""
# %% --- Bibliotecas y Funciones ---

from funciones import (
    # Bibliotecas
    plt,
    mne,
    sig,
    np,
    sns,
    stats,
    spectral_connectivity_epochs,
    nx,
    
    # Funciones
    EDF_to_RAW,
    welch,
    blackman_tukey,
    plot_periodograma,
    plotear_funcion_transferencia_wh,
    plotear_funcion_transferencia_ab,
    plotear_funcion_transferencia_sos,
    ancho_de_banda,
    filtrar_bandas,
    aplicar_filtros_ab,
    aplicar_filtros_sos,
    matriz_conectividad_PLI,
    limpiar_artefactos_ica,
    analisis_redes,
    binarizar_matriz_subrogados,
    procesar_y_extraer_subrogados
)

# %% --- Variables Globales ---
fs = 250
canales = np.array(['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2'])

# %% --- Conversión de archivos EDF a RAW ---
"""
h01_edf, h02, ... = sujetos saludables
s01, s02, ... = sujetos con esquizofrenia paranoide
"""
sujetos_sanos = {}
sujetos_con_esquizofrenia = {}
sujetos_h = np.array(['h01','h02','h03','h04','h05','h06','h07','h08','h09','h10','h11','h12','h13','h14'])
sujetos_s = np.array(['s01','s02','s03','s04','s05','s06','s07','s08','s09','s10','s11','s12','s13','s14'])

for i in range(1, 15):
    # El f"{i:02d}" le da el formato de 01, 02... 10, 11
    id_sano = f"h{i:02d}" 
    ruta_archivo_sano = f"Sujetos/{id_sano}.edf"
    id_esquizofrenia = f"s{i:02d}" 
    ruta_archivo_esquizofrenia = f"Sujetos/{id_esquizofrenia}.edf"
    
    sujetos_sanos[id_sano] = EDF_to_RAW(ruta_archivo_sano, sujeto=id_sano)
    sujetos_con_esquizofrenia[id_esquizofrenia] = EDF_to_RAW(ruta_archivo_esquizofrenia, sujeto=id_esquizofrenia)

# print(f"sujetos cargados correctamente: \n {list(sujetos_sanos.keys())} \n {list(sujetos_con_esquizofrenia.keys())}")
print("sujetos cargados correctamente")

# %% --- Visualización de la señal en dominio temporal ---
fig = sujetos_sanos['h01'].plot(duration=5, n_channels=19, scalings='auto', title='EEG Esquizofrenia')
plt.show()

ts_signal = sujetos_sanos['h01'].get_data()[0]
plt.figure()
plt.title('Señal EEG en el dominio Temporal', fontsize = 20)
plt.xlabel('Tiempo [s]', fontsize = 18)
plt.ylabel('Amplitud', fontsize = 18)
plt.plot(ts_signal, color = '#363636', label = 'Señal EEG: Sujeto H01 - Canal FP1')
plt.grid(True, which='both', ls=':')
plt.tick_params(axis='both', labelsize=16)
plt.xlim([75700, 76300])
plt.ylim([-2*10**(-5), 2*10**(-5)])
plt.legend(fontsize = 14)
plt.show()

# %% --- Estimación Espectral ANTES de Filtrar ---

sujeto = 'h01'
for i in range(19):
    """
    Estimación espectral de los 19 canales para un único sujeto
    """
    canal_idx = sujetos_sanos[sujeto].ch_names.index(canales[i])
    datos_canal = sujetos_sanos[sujeto].get_data()[canal_idx]
    
    f_welch, Px_welch = welch(cant_promedio = 10, vector = datos_canal, zp = 2, fs=fs, window ='hann')
    wb_welch = ancho_de_banda(f_welch, Px_welch, porcentaje = 99)
    #plot_periodograma(f_welch, Px_welch, wb_welch, title = f"h01 Welch - Canal {canales[i]}")
    
    # h01_bt, Px_h01_bt = blackman_tukey(datos_canal, fs = fs)
    # h01_wb = ancho_de_banda(sujetos_sanos['h01'], Px_h01_bt, porcentaje = 99)
    # plot_periodograma(h01_bt, Px_h01_bt, h01_wb, title = "h01 Blackman Tukey")

for i in range (14):
    '''
    Comparativa de los 14 sujetos sanos en un mismo canal
    '''
    canal_idx = sujetos_sanos[sujetos_h[i]].ch_names.index(canales[1])
    datos_canal = sujetos_sanos[sujetos_h[i]].get_data()[canal_idx]
    
    f_welch, Px_welch = welch(cant_promedio = 10, vector = datos_canal, zp = 2, fs=fs, window ='hann')
    wb_welch = ancho_de_banda(f_welch, Px_welch, porcentaje = 99)
    #plot_periodograma(f_welch, Px_welch, wb_welch, title = f"sujetos Sanos - Welch - Canal {canales[1]}")
     
for i in range (14):
    '''
    Comparativa de los 14 sujetos con esquizofrenia en un mismo canal
    '''
    canal_idx = sujetos_con_esquizofrenia[sujetos_s[i]].ch_names.index(canales[1])
    datos_canal = sujetos_con_esquizofrenia[sujetos_s[i]].get_data()[canal_idx]
    
    f_welch, Px_welch = welch(cant_promedio = 10, vector = datos_canal, zp = 2, fs=fs, window ='hann')
    wb_welch = ancho_de_banda(f_welch, Px_welch, porcentaje = 99)
    #plot_periodograma(f_welch, Px_welch, wb_welch, title = f"Sujeto con EP - Welch - Canal {canales[1]}")
    
# %% --- Diseño de Filtros IIR Butterworth ---

bandas_eeg = {
    'Delta': (2.0, 4.0),
    'Theta': (4.5, 7.5),
    'Alpha': (8.0, 12.5),
    'Beta':  (13.0, 30.0),
    'Gamma': (30.0, 45.0)
}

bandas = (['Delta','Theta','Alpha','Beta','Gamma'])

f_low_D, f_high_D = bandas_eeg['Delta']
f_low_T, f_high_T = bandas_eeg['Theta']
f_low_A, f_high_A = bandas_eeg['Alpha']
f_low_B, f_high_B = bandas_eeg['Beta']
f_low_G, f_high_G = bandas_eeg['Gamma']

# Diseño del filtro Butterworth de 2do orden
b_butter_D, a_butter_D = sig.butter(N=2, Wn=[f_low_D, f_high_D], btype='bandpass', fs=fs)
b_butter_T, a_butter_T = sig.butter(N=2, Wn=[f_low_T, f_high_T], btype='bandpass', fs=fs)
b_butter_A, a_butter_A = sig.butter(N=2, Wn=[f_low_A, f_high_A], btype='bandpass', fs=fs)
b_butter_B, a_butter_B = sig.butter(N=2, Wn=[f_low_B, f_high_B], btype='bandpass', fs=fs)
b_butter_G, a_butter_G = sig.butter(N=2, Wn=[f_low_G, f_high_G], btype='bandpass', fs=fs)

b_butter = [b_butter_D, b_butter_T, b_butter_A, b_butter_B, b_butter_G]
a_butter = [a_butter_D, a_butter_T, a_butter_A, a_butter_B, a_butter_G]

# Compruebo respuesta en frecuencia de los filtros
for i in range(5):
    plotear_funcion_transferencia_ab(b = b_butter[i], a = a_butter[i], fs=fs, label = f'{bandas[i]}', xlim =[0, 55], ylim = [-60, 1], f_banda = bandas_eeg[bandas[i]])

# %% --- Diseño de Filtros IIR chebyshev tipo II ---

ripple = 0.5 # dB
attenuation = 30 # dB
bdt = 0.7 # Ancho de la banda de transición
bdt_bp_izq = 0.2 # Distancia entre la frec. de corte y la frec. de paso
bdt_bp_der = 0.3

sos_cheby_D = sig.iirdesign(wp = [f_low_D + 0.1, f_high_D - 0.2], ws = [f_low_D - bdt, f_high_D + bdt], gpass = ripple, gstop = attenuation, 
                     analog = False, ftype = 'cheby2', output = 'sos', fs = fs)
sos_cheby_T = sig.iirdesign(wp = [f_low_T + bdt_bp_izq, f_high_T - bdt_bp_der], ws = [f_low_T - bdt, f_high_T + bdt], gpass = ripple, gstop = attenuation, 
                     analog = False, ftype = 'cheby2', output = 'sos', fs = fs)
sos_cheby_A = sig.iirdesign(wp = [f_low_A + bdt_bp_izq, f_high_A - bdt_bp_der], ws = [f_low_A - bdt, f_high_A + bdt], gpass = ripple, gstop = attenuation, 
                     analog = False, ftype = 'cheby2', output = 'sos', fs = fs)
sos_cheby_B = sig.iirdesign(wp = [f_low_B + bdt_bp_izq, f_high_B - bdt_bp_der], ws = [f_low_B - bdt, f_high_B + bdt], gpass = ripple, gstop = attenuation, 
                     analog = False, ftype = 'cheby2', output = 'sos', fs = fs)
sos_cheby_G = sig.iirdesign(wp = [f_low_G + bdt_bp_izq, f_high_G - bdt_bp_der], ws = [f_low_G - bdt, f_high_G + bdt], gpass = ripple, gstop = attenuation, 
                     analog = False, ftype = 'cheby2', output = 'sos', fs = fs)

sos_cheby = [sos_cheby_D, sos_cheby_T, sos_cheby_A, sos_cheby_B, sos_cheby_G]

for i in range(5):
    plotear_funcion_transferencia_sos(sos = sos_cheby[i], label = f'{bandas[i]}', fs=fs, f_banda = bandas_eeg[bandas[i]], xlim = [0, 55], ylim = [-60, 1])

# %% --- Aplico Filtros Butter en 1 canal ---

sujeto_prueba = 'h01'
canal_prueba = 'Fp1'
canal_idx = sujetos_sanos[sujeto_prueba].ch_names.index(canal_prueba)
datos_canal_crudos = sujetos_sanos[sujeto_prueba].get_data()[canal_idx]

num_muestras = len(datos_canal_crudos)
t = np.arange(num_muestras) / fs  # Vector de tiempo en segundos

# Ploteo el filtrado en las diferentes bandas para un mismo canal en el TIEMPO (no creo que lo use)
# plt.figure(2)
# plt.plot(t, datos_canal_crudos, label='Señal Cruda (con todos los ruidos y bandas)', color='lightgray')
# for i in range(5):
#     eeg_filtrado_banda = sig.filtfilt(b = b_butter[i], a = a_butter[i], x = datos_canal_crudos)
    
#     plt.plot(t, eeg_filtrado_banda, label=f'Señal Filtrada con filtro ({bandas[i]})', linewidth=1.5)
#     plt.title(f'Prueba de Filtrado en Dominio del Tiempo - Sujeto {sujeto_prueba}, Canal {canal_prueba}')
#     plt.xlabel('Tiempo [s]')
#     plt.ylabel('Amplitud [V]')
#     plt.legend()
#     plt.grid(True)

# Estimación espectral de la señal en crudo:
f, Px = welch(cant_promedio = 10, vector = datos_canal_crudos, zp = 2, fs=fs, window ='hann')
# Estimación del ancho de banda
wb = ancho_de_banda(f_welch, Px_welch, porcentaje = 90)

# Análisis espectral de las 5 bandas
plt.figure(2)
plt.plot(f, Px, label='PSD crudo', color='lightgray')
for i in range(5):
    eeg_filtrado_banda = sig.filtfilt(b = b_butter[i], a = a_butter[i], x = datos_canal_crudos)
    
    f, Px = welch(cant_promedio = 10, vector = eeg_filtrado_banda, zp = 2, fs=fs, window ='hann')
    
    plt.plot(f, Px, label=f'Filtrada con filtro ({bandas[i]})', linewidth=1.5)
    plt.title(f'Prueba de Filtrado en Dominio del Tiempo - Sujeto {sujeto_prueba}, Canal {canal_prueba}')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('PSD')
    plt.legend()
    plt.grid(True)
plt.xlim([0,45])

# %% --- Aplico Filtros Chebyshev en 1 canal---

# Análisis espectral de las 5 bandas
plt.figure(3)
plt.plot(f, Px, label='PSD crudo', color='lightgray')
for i in range(5):
    eeg_filtrado_banda = sig.sosfiltfilt(sos=sos_cheby[i], x = datos_canal_crudos)
    
    f, Px = welch(cant_promedio = 10, vector = eeg_filtrado_banda, zp = 2, fs=fs, window ='hann')
    
    plt.plot(f, Px, label=f'Filtrada con filtro ({bandas[i]})', linewidth=1.5)
    plt.title(f'Prueba de Filtrado en Dominio del Tiempo - Sujeto {sujeto_prueba}, Canal {canal_prueba}')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('PSD [dB]')
    plt.legend()
    plt.grid(True)
plt.xlim([0,45])

# %% --- Análisis espectral en una banda - Comparativa entre filtros ---
bnd = 0 # Eligo la banda que quiero plotear

f_low, f_high = bandas_eeg[bandas[bnd]]

eeg_filtrado_banda_butter = sig.filtfilt(b = b_butter[bnd], a = a_butter[bnd], x = datos_canal_crudos)
eeg_filtrado_banda_cheby = sig.sosfiltfilt(sos=sos_cheby[bnd], x = datos_canal_crudos)

f_crudo, Px_crudo = welch(cant_promedio = 10, vector = datos_canal_crudos, zp = 2, fs=fs, window ='hann')
f_butter, Px_butter = welch(cant_promedio = 10, vector = eeg_filtrado_banda_butter, zp = 2, fs=fs, window ='hann')
f_cheby, Px_cheby = welch(cant_promedio = 10, vector = eeg_filtrado_banda_cheby, zp = 2, fs=fs, window ='hann')

plt.plot(f_crudo, 10*np.log10(Px_crudo + 1e-15), label='PSD crudo', color='lightgray')
plt.plot(f_butter, 10*np.log10(Px_butter + 1e-15), label=f'Filtrada con filtro {bandas[bnd]} Butterworth', linewidth=1.5)
plt.plot(f_cheby, 10*np.log10(Px_cheby + 1e-15), label=f'Filtrada con filtro {bandas[bnd]} Chebyshev II', linewidth=1.5)
plt.axvline(x=f_low, color='red', linestyle=':', linewidth=2)
plt.axvline(x=f_high, color='red', linestyle=':', linewidth=2)
plt.title(f'Análisis espectral de PSD: Comparativa entre filtros - Banda {bandas[bnd]}', fontsize = 20)
plt.xlabel('Frecuencia [Hz]', fontsize = 18)
plt.ylabel('PSD [dB]', fontsize = 18)
plt.legend(fontsize = 14, loc = 'upper right')
plt.grid(True)
plt.tick_params(axis='both', labelsize=14)
plt.xlim([f_low-1, f_high+1])
plt.ylim([-150,-100])
plt.show()
    
# %% --- Análisis de Energías ---
energia_total_ecg_butter = np.sum(Px_butter)
energia_total_ecg_cheby = np.sum(Px_cheby)

print(f'Energía Butter: {energia_total_ecg_butter}\n')
print(f'Energía Cheby: {energia_total_ecg_cheby}\n')

# %% --- Comparativa en dominio del tiempo antes y despues de filtrar (no creo que valga la pena, no tiene sentido la comparación) ---
ts_signal = sujetos_sanos['h01'].get_data()[0]
plt.figure()
plt.title('Señal EEG en el dominio Temporal', fontsize = 20)
plt.xlabel('Tiempo [s]', fontsize = 18)
plt.ylabel('Amplitud', fontsize = 18)
plt.plot(ts_signal, color='lightgray', label = 'EEG sin filtrar')
plt.plot(eeg_filtrado_banda_butter, label = 'EEG filtrado con Butterworth')
plt.plot(eeg_filtrado_banda_cheby, label = 'EEG filtrado con Chebyshev')
plt.grid(True, which='both', ls=':')
plt.tick_params(axis='both', labelsize=16)
plt.xlim([75700, 76300])
plt.ylim([-2*10**(-5), 2*10**(-5)])
plt.legend(fontsize = 14)
plt.show()

# %% --- Elimino artefactos con ICA ---

sujetos_sanos_ICA = limpiar_artefactos_ica(sujetos_h, sujetos_sanos, fs=fs)
sujetos_esquizofrenia_ICA = limpiar_artefactos_ica(sujetos_s, sujetos_con_esquizofrenia, fs=fs)

# CAR (Common Average Reference):
for sujeto in sujetos_h:
    # Esto resta el promedio de los 19 canales a cada canal individual
    sujetos_sanos_ICA[sujeto].set_eeg_reference('average', projection=False, verbose=False)

for sujeto in sujetos_s:
    sujetos_esquizofrenia_ICA[sujeto].set_eeg_reference('average', projection=False, verbose=False)
    

# %% --- Aplico Filtros en los 19 canales para los 28 sujetos (532 señales) ---

# Filtro sujetos Sanos con Butterworth orden 2
print("\nAplicando filtro Butterworth en las señales...")

eeg_filtrado_sanos_butter = aplicar_filtros_ab(sujetos_h, sujetos_sanos_ICA, b_butter, a_butter, bandas)
eeg_filtrado_esquizofrenia_butter = aplicar_filtros_ab(sujetos_s, sujetos_esquizofrenia_ICA, b_butter, a_butter, bandas)

# Filtro sujetos Sanos con Chebyshev II
print("\nAplicando filtro Chebyshev tipo II en las señales...")

eeg_filtrado_sanos_cheby = aplicar_filtros_sos(lista_sujetos = sujetos_h, dict_crudo = sujetos_sanos_ICA, sos = sos_cheby, nombres_bandas = bandas)
eeg_filtrado_esquizofrenia_cheby = aplicar_filtros_sos(lista_sujetos = sujetos_s, dict_crudo = sujetos_esquizofrenia_ICA, sos = sos_cheby, nombres_bandas = bandas)

# %% --- Configuro banda de matrices de conectividad - Comparativa entre filtros ---
banda = 'Delta'
fmin, fmax = bandas_eeg[banda] # Frecuencias de corte de la banda

sujeto = 'h01' # Sujeto para las figuras de un unico sujeto

matriz_filtrada = eeg_filtrado_sanos_butter[sujeto][banda]

# Corto la señal continua en "Épocas" 
segundos_por_epoca = 5 # Definido a partir de la duración en segundos del EEG más corto: buscando tener más de 100 epocas por señal 
muestras_por_epoca = segundos_por_epoca * fs 

# %% --- Matríz de conectividad para 1 sujeto y 1 banda - Butter ---
print("\nCalculando Matriz de Conectividad PLI filtrado con Butter...")

n_epocas = matriz_filtrada.shape[1] // muestras_por_epoca # Cuántas épocas enteras entran en el registro
datos_recortados = matriz_filtrada[:, :n_epocas * muestras_por_epoca]

# Remodelamos de (19, N) a (19, n_epocas, muestras_por_epoca) y luego transponemos a (n_epocas, 19, muestras_por_epoca)
datos_3d = datos_recortados.reshape(19, n_epocas, muestras_por_epoca).transpose(1, 0, 2)
print(f"Señal dividida en {n_epocas} épocas de {segundos_por_epoca} segundos.")

# Phase-Lag Index (PLI): Compara la diferencia de fase entre los canales para medir su correlación 
resultado_conectividad = spectral_connectivity_epochs(
    data=datos_3d,
    method='pli',
    sfreq=fs,
    fmin=fmin,
    fmax=fmax,
    faverage=True, # Promedia las frecuencias de la banda en un solo valor por conexión
    verbose=False
)

# El argumento 'dense' devuelve la matriz cuadrada completa
matriz_adyacencia = resultado_conectividad.get_data(output='dense')[:, :, 0]
matriz_adyacencia_butter = matriz_adyacencia + matriz_adyacencia.T # Le sumo la transpuesta para que me grafique toda la matriz

# Visualización
plt.figure(figsize=(10, 8))
sns.heatmap(
    matriz_adyacencia_butter, 
    xticklabels=canales, 
    yticklabels=canales, 
    cmap='jet', 
    vmin=0, # El PLI va de 0 (nula sincronía con retraso) a 1 (sincronía perfecta)
    vmax=0.3, 
    square=True,
    linewidths=0.5
)

plt.title(f'Matriz de Adyacencia (PLI) - {sujeto} - Banda {banda} - Butter')
plt.xlabel('Canales')
plt.ylabel('Canales')
plt.show()

# %% --- Matríz de conectividad para 1 sujeto y 1 banda - Cheby ---
print("\nCalculando Matriz de Conectividad PLI filtrado con Cheby...")

matriz_filtrada = eeg_filtrado_sanos_cheby[sujeto][banda]

n_epocas = matriz_filtrada.shape[1] // muestras_por_epoca
datos_recortados = matriz_filtrada[:, :n_epocas * muestras_por_epoca]

# Remodelamos de (19, N) a (19, n_epocas, 500) y luego transponemos a (n_epocas, 19, 500)
datos_3d = datos_recortados.reshape(19, n_epocas, muestras_por_epoca).transpose(1, 0, 2)
print(f"Señal dividida en {n_epocas} épocas de {segundos_por_epoca} segundos.")

# Phase-Lag Index (PLI)
resultado_conectividad = spectral_connectivity_epochs(
    data=datos_3d,
    method='pli',
    sfreq=fs,
    fmin=fmin,
    fmax=fmax,
    faverage=True, # Promedia las frecuencias de la banda en un solo valor por conexión
    verbose=False
)

matriz_adyacencia = resultado_conectividad.get_data(output='dense')[:, :, 0]
matriz_adyacencia_cheby = matriz_adyacencia + matriz_adyacencia.T # Le sumo la transpuesta para que me grafique toda la matriz

# Visualización
plt.figure(figsize=(10, 8))
sns.heatmap(
    matriz_adyacencia_cheby, 
    xticklabels=canales, 
    yticklabels=canales, 
    cmap='jet', 
    vmin=0, # El PLI va de 0 (nula sincronía con retraso) a 1 (sincronía perfecta)
    vmax=0.3, 
    square=True,
    linewidths=0.5
)

plt.title(f'Matriz de Adyacencia (PLI) - {sujeto} - Banda {banda} - Chebyshev tipo II')
plt.xlabel('Canales')
plt.ylabel('Canales')
plt.show()

# %% --- Visualización de matrices: Sujeto H01 con Butter vs Cheby en una unica banda de frecuencias ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Heatmap Sanos
sns.heatmap(
    matriz_adyacencia_butter, xticklabels=canales, yticklabels=canales, 
    cmap='jet', vmin=0, vmax=0.3, square=True, linewidths=0.5, ax=axes[0]
)
axes[0].set_title(f'Matriz de Conectividad - Sujeto H01 - PLI {banda} - Butter', fontsize = 16)
axes[0].set_xlabel('Canales', fontsize = 14)
axes[0].set_ylabel('Canales', fontsize = 14)

# Heatmap Esquizofrenia
sns.heatmap(
    matriz_adyacencia_cheby, xticklabels=canales, yticklabels=canales, 
    cmap='jet', vmin=0, vmax=0.3, square=True, linewidths=0.5, ax=axes[1]
)
axes[1].set_title(f'Matriz de Conectividad - Sujeto H01 - PLI {banda} - Cheby', fontsize = 16)
axes[1].set_xlabel('Canales', fontsize = 14)
axes[1].set_ylabel('Canales', fontsize = 14)

plt.tight_layout()
plt.show()

# %% --- Matriz de conectividad para 14 Sanos vs 14 Esquizofrenia en 1 banda - Butter ---

matrices_PLI_sanos_butter = matriz_conectividad_PLI(sujetos_h, eeg_filtrado_sanos_butter, banda, fmin, fmax, segundos_epoca=segundos_por_epoca)
matrices_PLI_esquizofrenia_butter = matriz_conectividad_PLI(sujetos_s, eeg_filtrado_esquizofrenia_butter, banda, fmin, fmax, segundos_epoca=segundos_por_epoca)

# %% --- Visualización de un cerebro 'promedio' sano vs esquizofrenia - Butter ---
promedio_sanos_butter = np.mean(matrices_PLI_sanos_butter, axis=0)
promedio_esquizofrenia_butter = np.mean(matrices_PLI_esquizofrenia_butter, axis=0)

# Visualización
print("Generando gráficos comparativos...")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Heatmap Sanos
sns.heatmap(
    promedio_sanos_butter, xticklabels=canales, yticklabels=canales, 
    cmap='jet', vmin=0, vmax=0.3, square=True, linewidths=0.5, ax=axes[0]
)
axes[0].set_title(f'Promedio Sanos (n=14) - PLI {banda} - Butter')
axes[0].set_xlabel('Canales')
axes[0].set_ylabel('Canales')

# Heatmap Esquizofrenia
sns.heatmap(
    promedio_esquizofrenia_butter, xticklabels=canales, yticklabels=canales, 
    cmap='jet', vmin=0, vmax=0.3, square=True, linewidths=0.5, ax=axes[1]
)
axes[1].set_title(f'Promedio Esquizofrenia (n=14) - PLI {banda} - Butter')
axes[1].set_xlabel('Canales')
axes[1].set_ylabel('Canales')

plt.tight_layout()
plt.show()

# %% --- Matriz de conectividad para 14 Sanos vs 14 Esquizofrenia en 1 banda - Chebyshev II ---

matrices_PLI_sanos_cheby = matriz_conectividad_PLI(sujetos_h, eeg_filtrado_sanos_cheby, banda, fmin, fmax, segundos_epoca = segundos_por_epoca)
matrices_PLI_esquizofrenia_cheby = matriz_conectividad_PLI(sujetos_s, eeg_filtrado_esquizofrenia_cheby, banda, fmin, fmax, segundos_epoca = segundos_por_epoca)

# %% --- Visualización de un cerebro 'promedio' sano vs esquizofrenia - Chebyshev II ---

promedio_sanos_cheby = np.mean(matrices_PLI_sanos_cheby, axis=0)
promedio_esquizofrenia_cheby = np.mean(matrices_PLI_esquizofrenia_cheby, axis=0)

# 4. Visualización Comparativa (Sanos vs Esquizofrenia)
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Heatmap Sanos
sns.heatmap(
    promedio_sanos_cheby, xticklabels=canales, yticklabels=canales, 
    cmap='jet', vmin=0, vmax=0.3, square=True, linewidths=0.5, ax=axes[0]
)
axes[0].set_title(f'Promedio Sanos (n=14) - PLI {banda} - Chebyshev tipo II')
axes[0].set_xlabel('Canales')
axes[0].set_ylabel('Canales')

# Heatmap Esquizofrenia
sns.heatmap(
    promedio_esquizofrenia_cheby, xticklabels=canales, yticklabels=canales, 
    cmap='jet', vmin=0, vmax=0.3, square=True, linewidths=0.5, ax=axes[1]
)
axes[1].set_title(f'Promedio Esquizofrenia (n=14) - PLI {banda} - Chebyshev tipo II')
axes[1].set_xlabel('Canales')
axes[1].set_ylabel('Canales')

plt.tight_layout()
plt.show()

# %% --- Análisis de Redes para 1 banda de frecuencias tomando 30% de uniones más fuertes ---
print(f">> Realizando analisis de redes para la banda {banda}")

datos_sujetos = {} # Aca voy a guardar los datos del análisis para todos los sujetos

# Matrices binarizadas separadas por filtro utilizado y población
matrices_PLI_sanos_butter_bin = {} 
matrices_PLI_esquizofrenia_butter_bin = {}
matrices_PLI_sanos_cheby_bin = {}
matrices_PLI_esquizofrenia_cheby_bin = {}

posiciones_electrodos = {
    'Fp1': (-1.5, 3), 'Fp2': (1.5, 3),
    'F7':  (-3, 2),   'F3':  (-1.5, 2), 'Fz': (0, 2),   'F4': (1.5, 2),  'F8': (3, 2),
    'T3':  (-3, 0),   'C3':  (-1.5, 0), 'Cz': (0, 0),   'C4': (1.5, 0),  'T4': (3, 0),
    'T5':  (-3, -2),  'P3':  (-1.5, -2),'Pz': (0, -2),  'P4': (1.5, -2), 'T6': (3, -2),
    'O1':  (-1.5, -3),'O2':  (1.5, -3)
} # Sirve para plotear los grafos con la misma disposición que los electrodos

eje_x = len(canales)

for i in range(14):
    M_bin, G, Edges, BC, Clust = analisis_redes(matrices_PLI_sanos_butter[i])
    
    datos_sujetos[f'Sujeto_H{i+1:02d}_Butter'] = {
        'M_bin': M_bin,
        'G': G,
        'Edges': Edges,
        'BC': BC,
        'Clust': Clust,
    }
    
    print(f"Sujeto H{i+1:02d} - Butter procesado: {Edges} enlaces")
    
    M_bin, G, Edges, BC, Clust = analisis_redes(matrices_PLI_sanos_cheby[i])
    
    datos_sujetos[f'Sujeto_H{i+1:02d}_Cheby'] = {
        'M_bin': M_bin,
        'G': G,
        'Edges': Edges,
        'BC': BC,
        'Clust': Clust,
    }
    
    print(f"Sujeto H{i+1:02d} - Cheby procesado: {Edges} enlaces")
    
    M_bin, G, Edges, BC, Clust = analisis_redes(matrices_PLI_esquizofrenia_butter[i])
    
    datos_sujetos[f'Sujeto_S{i+1:02d}_Butter'] = {
        'M_bin': M_bin,
        'G': G,
        'Edges': Edges,
        'BC': BC,
        'Clust': Clust,
    }
    
    print(f"Sujeto S{i+1:02d} - Butter procesado: {Edges} enlaces")
    
    M_bin, G, Edges, BC, Clust = analisis_redes(matrices_PLI_esquizofrenia_cheby[i])
    
    datos_sujetos[f'Sujeto_S{i+1:02d}_Cheby'] = {
        'M_bin': M_bin,
        'G': G,
        'Edges': Edges,
        'BC': BC,
        'Clust': Clust,
    }
    
    print(f"Sujeto S{i+1:02d} - Cheby procesado: {Edges} enlaces")
    
    # Llaves para tomar los datos dentro del diccionario 'datos_sujetos'
    llave_paciente_h_butter = f'Sujeto_H{i+1:02d}_Butter'
    llave_paciente_h_cheby = f'Sujeto_H{i+1:02d}_Cheby'
    llave_paciente_s_butter = f'Sujeto_S{i+1:02d}_Butter'
    llave_paciente_s_cheby = f'Sujeto_S{i+1:02d}_Cheby'
    
    Edges_h_butter = datos_sujetos[llave_paciente_h_butter]['Edges']
    Edges_h_cheby = datos_sujetos[llave_paciente_h_cheby]['Edges']
    Edges_s_butter = datos_sujetos[llave_paciente_s_butter]['Edges']
    Edges_s_cheby = datos_sujetos[llave_paciente_s_cheby]['Edges']
    
    print(f'Para el sujeto H{i+1:02d}, la cantidad de enlaces es de: {Edges_h_butter} enlaces con Butter y {Edges_h_cheby} enlaces con Cheby')
    print(f'Para el sujeto S{i+1:02d}, la cantidad de enlaces es de: {Edges_s_butter} enlaces con Butter y {Edges_s_cheby} enlaces con Cheby')
    
    # # %% --- Visualización de Grafos ---
    
    # # Agarro los grafos de los sujetos de control
    # G_h_butter = datos_sujetos[llave_paciente_h_butter]['G']
    # G_h_cheby = datos_sujetos[llave_paciente_h_cheby]['G']
    
    # # Sujetos con esquizofrenia
    # G_s_butter = datos_sujetos[llave_paciente_s_butter]['G']
    # G_s_cheby = datos_sujetos[llave_paciente_s_cheby]['G']
    
    # # Configuracion para las posiciones de los nodos
    # mapeo_nombres = {j: canales[j] for j in range(len(canales))}
    
    # G_h_butter_etiq = nx.relabel_nodes(G_h_butter, mapeo_nombres)
    # G_h_cheby_etiq = nx.relabel_nodes(G_h_cheby, mapeo_nombres)
    # G_s_butter_etiq = nx.relabel_nodes(G_s_butter, mapeo_nombres)
    # G_s_cheby_etiq = nx.relabel_nodes(G_s_cheby, mapeo_nombres)
    
    # # FIGURA 1: Sujeto H'XX'
    # fig_h, (ax1_h, ax2_h) = plt.subplots(1, 2, figsize=(14, 7))
    # fig_h.suptitle(f"Comparativa de Redes Funcionales - Sujeto Control H{i+1:02d}", fontsize=18)
    
    # # Subplot Izquierdo: Butterworth
    # nx.draw_networkx(
    #     G_h_butter_etiq, pos=posiciones_electrodos, ax=ax1_h,
    #     with_labels=True, node_color='#4F81BD', node_size=900,
    #     font_size=10, edge_color='gray', width=1.5
    # )
    # ax1_h.set_title("Filtro Butterworth", fontsize=15)
    # ax1_h.axis('off')
    # ax1_h.set_xlim(-4, 4); ax1_h.set_ylim(-4, 3.5)

    # # Subplot Derecho: Chebyshev II
    # nx.draw_networkx(
    #     G_h_cheby_etiq, pos=posiciones_electrodos, ax=ax2_h,
    #     with_labels=True, node_color='#C0504D', node_size=900,
    #     font_size=10, edge_color='gray', width=1.5
    # )
    # ax2_h.set_title("Filtro Chebyshev II", fontsize=15)
    # ax2_h.axis('off')
    # ax2_h.set_xlim(-4, 4); ax2_h.set_ylim(-4, 3.5)

    # plt.tight_layout()
    # plt.show()

    # # FIGURA 2: Sujeto con esquizofrenia
    # fig_s, (ax1_s, ax2_s) = plt.subplots(1, 2, figsize=(14, 7))
    # fig_s.suptitle(f"Comparativa de Redes Funcionales - Sujeto Esquizofrenia S{i+1:02d}", fontsize=18)
    
    # # Subplot Izquierdo: Butterworth
    # nx.draw_networkx(
    #     G_s_butter_etiq, pos=posiciones_electrodos, ax=ax1_s,
    #     with_labels=True, node_color='#4F81BD', node_size=900,
    #     font_size=10, edge_color='gray', width=1.5
    # )
    # ax1_s.set_title("Filtro Butterworth", fontsize=15)
    # ax1_s.axis('off')
    # ax1_s.set_xlim(-4, 4); ax1_s.set_ylim(-4, 3.5)

    # # Subplot Derecho: Chebyshev II 
    # nx.draw_networkx(
    #     G_s_cheby_etiq, pos=posiciones_electrodos, ax=ax2_s,
    #     with_labels=True, node_color='#C0504D', node_size=900,
    #     font_size=10, edge_color='gray', width=1.5
    # )
    # ax2_s.set_title("Filtro Chebyshev II", fontsize=15)
    # ax2_s.axis('off')
    # ax2_s.set_xlim(-4, 4); ax2_s.set_ylim(-4, 3.5)

    # plt.tight_layout()
    # plt.show()
    

# %% --- Visualización de matrices binarizadas (para 1 sujeto y 1 banda) ---
# sujeto = 3
# filtro = 'Cheby'
# llave = f'Sujeto_H{sujeto:02d}_{filtro}'
# plt.figure(figsize=(10, 8))
# sns.heatmap(
#     data=datos_sujetos[llave]['M_bin'], 
#     xticklabels=canales, 
#     yticklabels=canales, 
#     cmap='jet', 
#     vmin=0, # El PLI va de 0 (nula sincronía con retraso) a 1 (sincronía perfecta)
#     vmax=1, 
#     square=True,
#     linewidths=0.5
# )
# plt.title('Matriz de Adyacencia (PLI) Binarizada - Chebyshev tipo II')
# plt.xlabel('Canales')
# plt.ylabel('Canales')
# plt.show()

# # %% --- Visualización Grafos (para todos los sujetos de una banda y 1 población) ---
# mapeo_nombres = {i: canales[i] for i in range(len(canales))}

# i = 0
# llave_paciente = f'Paciente_sano_{i}'

# G = analisis_redes[llave_paciente]['G']

# G_etiquetado = nx.relabel_nodes(G, mapeo_nombres)

# plt.figure(figsize=(8, 8))
# plt.title(f'Red de Conectividad Funcional - Sujeto H{i+1:02d}', fontsize=19)

# nx.draw_networkx(
#     G_etiquetado,
#     pos=posiciones_electrodos,
#     with_labels=True,
#     node_color='lightblue',
#     node_size=1000,
#     font_size=10,
#     font_weight='bold',
#     edge_color='gray',
#     width=1.5, # Grosor de las conexiones
#     alpha=0.9
# )

# plt.axis('off')

# plt.xlim(-4, 4)
# plt.ylim(-4, 4)

# plt.tight_layout()
# plt.show()


# %% --- Análisis Estadístico con método de análisis de datos subrogados ---
datos_sujetos_estadisticos = {}

for i in range(14):

    llave_h = f'h{i+1:02d}'
    llave_s = f's{i+1:02d}'
    
    print(f"\n--- Analizando Sujeto H{i+1:02d} ---")
    
    # Sano - Butterworth
    M_bin, P_val, G, Edges, BC, Clust = procesar_y_extraer_subrogados(
        eeg_filtrado_sanos_butter[llave_h], banda, matrices_PLI_sanos_butter[i], fs, fmin, fmax, muestras_por_epoca
    )
    datos_sujetos_estadisticos[f'Sujeto_H{i+1:02d}_Butter'] = {'M_bin': M_bin, 'p_values': P_val, 'G': G, 'Edges': Edges, 'BC': BC, 'Clust': Clust}
    print(f" > Butter procesado: {Edges} enlaces significativos")
    
    # Sano - Chebyshev
    M_bin, P_val, G, Edges, BC, Clust = procesar_y_extraer_subrogados(
        eeg_filtrado_sanos_cheby[llave_h], banda, matrices_PLI_sanos_cheby[i], fs, fmin, fmax, muestras_por_epoca
    )
    datos_sujetos_estadisticos[f'Sujeto_H{i+1:02d}_Cheby'] = {'M_bin': M_bin, 'p_values': P_val, 'G': G, 'Edges': Edges, 'BC': BC, 'Clust': Clust}
    print(f" > Cheby procesado: {Edges} enlaces significativos")
    
    print(f"--- Analizando Sujeto S{i+1:02d} ---")
    
    # Esquizofrenia - Butterworth
    M_bin, P_val, G, Edges, BC, Clust = procesar_y_extraer_subrogados(
        eeg_filtrado_esquizofrenia_butter[llave_s], banda, matrices_PLI_esquizofrenia_butter[i], fs, fmin, fmax, muestras_por_epoca
    )
    datos_sujetos_estadisticos[f'Sujeto_S{i+1:02d}_Butter'] = {'M_bin': M_bin, 'p_values': P_val, 'G': G, 'Edges': Edges, 'BC': BC, 'Clust': Clust}
    print(f" > Butter procesado: {Edges} enlaces significativos")
    
    # Esquizofrenia - Chebyshev
    M_bin, P_val, G, Edges, BC, Clust = procesar_y_extraer_subrogados(
        eeg_filtrado_esquizofrenia_cheby[llave_s], banda, matrices_PLI_esquizofrenia_cheby[i], fs, fmin, fmax, muestras_por_epoca
    )
    datos_sujetos_estadisticos[f'Sujeto_S{i+1:02d}_Cheby'] = {'M_bin': M_bin, 'p_values': P_val, 'G': G, 'Edges': Edges, 'BC': BC, 'Clust': Clust}
    print(f" > Cheby procesado: {Edges} enlaces significativos")

# %% --- VISUALIZACIÓN DE GRAFOS ---
# Configuracion para las posiciones de los nodos
mapeo_nombres = {j: canales[j] for j in range(len(canales))}
sujeto = 0
G_h_butter = list(datos_sujetos_estadisticos[f'Sujeto_H{sujeto+1:02d}_Butter']['G'].values())
G_h_cheby = list(datos_sujetos_estadisticos[f'Sujeto_H{sujeto+1:02d}_Cheby']['G'].values())
G_s_butter = list(datos_sujetos_estadisticos[f'Sujeto_S{sujeto+1:02d}_Butter']['G'].values())
G_s_cheby = list(datos_sujetos_estadisticos[f'Sujeto_S{sujeto+1:02d}_Cheby']['G'].values())

G_h_butter_etiq = nx.relabel_nodes(G_h_butter, mapeo_nombres)
G_h_cheby_etiq = nx.relabel_nodes(G_h_cheby, mapeo_nombres)
G_s_butter_etiq = nx.relabel_nodes(G_s_butter, mapeo_nombres)
G_s_cheby_etiq = nx.relabel_nodes(G_s_cheby, mapeo_nombres)

# FIGURA 1: Sujeto H'XX'
fig_h, (ax1_h, ax2_h) = plt.subplots(1, 2, figsize=(14, 7))
fig_h.suptitle(f"Comparativa de Redes Funcionales - Sujeto Control H{i+1:02d}", fontsize=18)

# Subplot Izquierdo: Butterworth
nx.draw_networkx(
    G_h_butter_etiq, pos=posiciones_electrodos, ax=ax1_h,
    with_labels=True, node_color='#4F81BD', node_size=900,
    font_size=10, edge_color='gray', width=1.5
)
ax1_h.set_title("Filtro Butterworth", fontsize=15)
ax1_h.axis('off')
ax1_h.set_xlim(-4, 4); ax1_h.set_ylim(-4, 3.5)

# Subplot Derecho: Chebyshev II
nx.draw_networkx(
    G_h_cheby_etiq, pos=posiciones_electrodos, ax=ax2_h,
    with_labels=True, node_color='#C0504D', node_size=900,
    font_size=10, edge_color='gray', width=1.5
)
ax2_h.set_title("Filtro Chebyshev II", fontsize=15)
ax2_h.axis('off')
ax2_h.set_xlim(-4, 4); ax2_h.set_ylim(-4, 3.5)

plt.tight_layout()
plt.show()

# FIGURA 2: Sujeto con esquizofrenia
fig_s, (ax1_s, ax2_s) = plt.subplots(1, 2, figsize=(14, 7))
fig_s.suptitle(f"Comparativa de Redes Funcionales - Sujeto Esquizofrenia S{i+1:02d}", fontsize=18)

# Subplot Izquierdo: Butterworth
nx.draw_networkx(
    G_s_butter_etiq, pos=posiciones_electrodos, ax=ax1_s,
    with_labels=True, node_color='#4F81BD', node_size=900,
    font_size=10, edge_color='gray', width=1.5
)
ax1_s.set_title("Filtro Butterworth", fontsize=15)
ax1_s.axis('off')
ax1_s.set_xlim(-4, 4); ax1_s.set_ylim(-4, 3.5)

# Subplot Derecho: Chebyshev II 
nx.draw_networkx(
    G_s_cheby_etiq, pos=posiciones_electrodos, ax=ax2_s,
    with_labels=True, node_color='#C0504D', node_size=900,
    font_size=10, edge_color='gray', width=1.5
)
ax2_s.set_title("Filtro Chebyshev II", fontsize=15)
ax2_s.axis('off')
ax2_s.set_xlim(-4, 4); ax2_s.set_ylim(-4, 3.5)

plt.tight_layout()
plt.show()

print("\n>> ¡Análisis de subrogados completado exitosamente!")

# %% --- Duracion de las imagenes EEG en tiempo (para definir la cantidad de segundos por época) ---
duraciones_h = []
duraciones_s = []

for i in range(1, 15):
    id_h = f'h{i:02d}'
    id_s = f's{i:02d}'
    
    duraciones_h.append(sujetos_sanos[id_h].times[-1])
    duraciones_s.append(sujetos_con_esquizofrenia[id_s].times[-1])

# --- Visualización ---
plt.figure(figsize=(14, 6))
x = np.arange(14)
width = 0.35

plt.bar(x - width/2, duraciones_h, width, label='Control (Sanos)', color='lightblue', edgecolor='black')
plt.bar(x + width/2, duraciones_s, width, label='Esquizofrenia', color='lightcoral', edgecolor='black')

plt.title('Duración Total de los Registros EEG por Sujeto', fontsize=18, fontweight='bold')
plt.ylabel('Duración (Segundos)', fontsize=14)
plt.xlabel('Número de Sujeto (01 al 14)', fontsize=14)

plt.xticks(x, [f'{i+1:02d}' for i in range(14)], fontsize=12)

plt.axhline(300, color='red', linestyle='--', alpha=0.7, label='5 Minutos (300 s)')
plt.axhline(900, color='green', linestyle='--', alpha=0.7, label='15 Minutos (900 s)')

plt.legend(fontsize=12, loc='upper right')
plt.grid(axis='y', ls=':', alpha=0.7)
plt.tight_layout()
plt.show()

# --- Consola ---
min_h = min(duraciones_h)
min_s = min(duraciones_s)
min_absoluto = min(min_h, min_s)

print(f"Duración máxima (Sanos): {max(duraciones_h):.2f} s")
print(f"Duración máxima (Esquizofrenia): {max(duraciones_s):.2f} s")
print("-" * 30)
print(f"Duración mínima (Sanos): {min_h:.2f} s")
print(f"Duración mínima (Esquizofrenia): {min_s:.2f} s")
print("-" * 30)
print(f">> EL REGISTRO MÁS CORTO DURA: {min_absoluto:.2f} segundos.")