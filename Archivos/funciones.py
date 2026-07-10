# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 22:03:52 2026

@author: Tomás Altimare Bercovich

Descripción:
    
"""

#%%##############
## Bibliotecas ##
#%%##############
import os # Se usa en EDF_to_RAW
import mne # https://mne.tools/stable/index.html
from mne_connectivity import spectral_connectivity_epochs
from mne.preprocessing import ICA # Para eliminar los artefactos (principalmente ruidos por pestañeo)
from mne.viz import plot_filter
from mne.filter import create_filter
import seaborn as sns # https://seaborn.pydata.org/generated/seaborn.heatmap.html
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.stats as stats
import numpy as np
from pytc2.sistemas_lineales import plot_plantilla # Plantilla de diseño para filtros FIR 
from matplotlib import patches # Para poder hacer el diagrama de polos y ceros de los filtros

#%%############
## Funciones ##
#%%############

def EDF_to_RAW (edf_file, sujeto):
    """
    Descripción: Pasa de un archivo EDF a un archivo RAW
    -----------
    Notas:
    -----
    # https://mne.tools/stable/generated/mne.io.read_raw_edf.html
    # Se utiliza preload = True para poder filtrar después
    """
   
    print(f"Cargando {edf_file}...")
    try:
        raw_file = mne.io.read_raw_edf(edf_file, preload=True)
    except FileNotFoundError:
        print(f"ERROR: El archivo '{edf_file}' no fue encontrado.")
        print(f"Asegúrate de que esté en: {os.getcwd()}")
        # Paramos el script si no hay archivo
        raise
    
    print(f"\n--- Información del registro {sujeto} ---")
    print(raw_file.info)
    print(f"\nFrecuencia de muestreo: {raw_file.info['sfreq']} Hz")
    print(f"Canales: {raw_file.ch_names}")
    return raw_file

def welch(cant_promedio,vector,zp,fs,window):
    N = vector.shape[0] # N=Cantidad de muestras y accede al elemento 0
    nperseg = N//cant_promedio # L=largo del bloque
    print(f'El largo del bloque es: {nperseg}\n')
    nfft = zp*nperseg
    # Periodograma
    f_w, Px_w = sig.welch(vector, fs=fs, nperseg=nperseg, window=window, nfft=nfft)
    return f_w, Px_w

def blackman_tukey(x,  M = None, fs = 1000):    
    # N = len(x)
    x_z = x.shape
    N = np.max(x_z)
    df = fs/N
    
    if M is None:
        M = N//5 # N/5 < M < N, por lo que si no selecciono nada, elijo N/5
    
    r_len = (2*M) - 1

    xx = x.ravel()[:r_len];

    # Correlaciono la señal para no tomar los valores de los "bordes"
    r = np.correlate(xx, xx, mode='same') / r_len

    Px_bt = np.abs(np.fft.fft(r * sig.windows.blackman(r_len), n = N) )
    Px_bt = Px_bt.reshape(x_z)
    
    f_bt = np.linspace(0, (N-1)*df, N)
    
    return f_bt, Px_bt;

def plot_periodograma(f,Px,f_wb,title, f0 = 0):
    """
    Plotea un periodograma. La linea roja indica el ancho de banda.
    """
    plt.figure()
    plt.plot(f,Px)
    plt.axvline(f_wb, color='r', linestyle='--', label=f'BW ≈ {f_wb} Hz')
    plt.title(title)
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("PSD") # Power Spectral Density (or power spectrum of x)
    #plt.xlim([f0,50])
    plt.xlim([f0,f_wb + 1])
    plt.grid()
    return

def plotear_funcion_transferencia_wh(w, h, fpass, fstop,
                                  ripple = 1, attenuation = 40,
                                  xlim = (0, 10), label = 'T(s)', 
                                  filter_type = 'bandpass', xlim_s = False,
                                  fig_inicial = 0, fs = 250):
    """
        Plotea la respuesta en frecuencia de una función transferencia de:
            - Módulo
            - Fase
            - Retardo de grupo
            - Polos y Ceros
        
        b: Vector de coeficientes que modifican la entrada del sistema
        a: Vector de coeficientes que modifican la salida del sistema
        xlim: Limites horizontales del gráfico
        xlim_s: Status del xlim (True si quiero limitar el gráfico, 
                                 False si quiero imprimirlo sin limites)
        fig_inicial: Número que quiero que tenga la primer figura a 
                    imprimir. Sirve para sobre-escribir (o no) plotteos 
                    de diferentes funciones en un mismo gráfico (dejar en 
                    0 si se quiere sobre-plotear)
    """
    
    phase = np.unwrap(np.angle(h)) # fase del grupo
    gd = -np.diff(phase) / np.diff(w) # retardo
   
    # --- Ploteo ---
    figura = 1
    # Magnitud
    plt.figure(fig_inicial + figura)
    plt.title(f'Figura {fig_inicial + figura}: Respuesta en Magnitud')
    plt.xlabel('Pulsación angular [r/s]')
    plt.ylabel('|H(jω)| [dB]')
    plt.plot(w, 20*np.log10(abs(h)), label = f"{label}")
    if xlim_s == True: plt.xlim(xlim)
    plt.grid(True, which='both', ls=':')
    plot_plantilla(filter_type = filter_type, fpass = fpass, ripple = ripple, fstop = fstop, attenuation = attenuation, fs = fs)
    plt.legend()
    figura+=1

    # Fase
    plt.figure(fig_inicial + figura)
    plt.title(f'Figura {fig_inicial + figura}: Respuesta de Fase')
    plt.xlabel('Pulsación angular [r/s]')
    plt.ylabel('Fase [°]')
    plt.plot(w, np.degrees(phase), label = f"{label}")
    if xlim_s == True: plt.xlim(xlim)
    plt.grid(True, which='both', ls=':')
    plt.legend()
    figura+=1
    
    # Retardo de grupo
    plt.figure(fig_inicial + figura)
    plt.title(f'Figura {fig_inicial + figura}: Retardo de Grupo')
    plt.xlabel('Pulsación angular [r/s]')
    plt.plot(w[:-1], gd, label = f"{label}")
    plt.ylabel('τg [s]')
    if xlim_s == True: plt.xlim(xlim)
    plt.grid(True, which='both', ls=':')
    plt.legend()
    figura+=1

def plotear_funcion_transferencia_ab(b, a, fs, xlim = None, ylim = None, label = 'T(s)', polos_y_ceros = True, magnitud_fase_retardo = True, fig_inicial = 0, f_banda = None, fs_title=20, fs_label=18, fs_legend=14, fs_ticks=14):
    """
        Plotea la respuesta en frecuencia de una función transferencia de:
            - Módulo
            - Fase
            - Retardo de grupo
            - Polos y Ceros
        
        b: Vector de coeficientes que modifican la entrada del sistema
        a: Vector de coeficientes que modifican la salida del sistema
        xlim: Limites horizontales del gráfico
        xlim_s: Status del xlim (True si quiero limitar el gráfico, 
                                 False si quiero imprimirlo sin limites)
        fig_inicial: Número que quiero que tenga la primer figura a 
                    imprimir. Sirve para sobre-escribir (o no) plotteos 
                    de diferentes funciones en un mismo gráfico (dejar en 
                    0 si se quiere sobre-plotear)
        f_banda: Tupla con los límites de la banda a sombrear (ej: (8.0, 12.5))
        fs_title: Tamaño de la fuente para los títulos.
        fs_label: Tamaño de la fuente para las etiquetas de los ejes.
        fs_legend: Tamaño de la fuente para las leyendas.
        fs_ticks: Tamaño de la fuente para los números de los ejes.
    """
    # --- Calculo polos y ceros ---
    z, p, k = sig.tf2zpk(b, a) # Zpk = [ [z0,z1,...,zn], [p0,p1,...,pn], k]

    # --- Módulo, Fase y Retardo de Fase ---
    f, h = sig.freqz(b, a, fs=fs) # worN = np.logspace(2,-2,1000) (dominio Z)
    phase = np.unwrap(np.angle(h)) # fase del grupo
    gd = -np.diff(phase) / (2 * np.pi * np.diff(f))
    
    # --- Ploteo ---
    figura = 1
    #if magnitud_fase_retardo == True:
        # Magnitud
    plt.figure(fig_inicial + figura)
    plt.title('Respuesta en Magnitud', fontsize=fs_title)
    plt.xlabel('Frecuencia [Hz]', fontsize=fs_label)
    plt.ylabel('|H(f)| [dB]', fontsize=fs_label)
    
    curva = plt.plot(f, 20*np.log10(abs(h)), label = f"Filtro {label}") 
    color_curva = curva[0].get_color()
    
    if f_banda is not None: 
        plt.axvspan(f_banda[0], f_banda[1], color=color_curva, alpha=0.15, label=f'Banda {label}', zorder=0)
        
        ticks_actuales = plt.gca().get_xticks() 
        nuevos_ticks = list(ticks_actuales) + [f_banda[0], f_banda[1]]
        nuevos_ticks = sorted(list(set(nuevos_ticks))) 
        
        if xlim is not None:
            nuevos_ticks = [t for t in nuevos_ticks if xlim[0] <= t <= xlim[1]]
            
        etiquetas_escalonadas = []
        nivel_abajo = False
        
        for i in range(len(nuevos_ticks)):
            if i > 0 and (nuevos_ticks[i] - nuevos_ticks[i-1]) < 2.0:
                nivel_abajo = not nivel_abajo 
            else:
                nivel_abajo = False
                
            texto = f"{nuevos_ticks[i]:g}" 
            
            if nivel_abajo:
                etiquetas_escalonadas.append(f"\n{texto}")
            else:
                etiquetas_escalonadas.append(texto)
                
        plt.xticks(nuevos_ticks, etiquetas_escalonadas, rotation=0)
            
        plt.tick_params(axis='both', labelsize=fs_ticks)
        
        if xlim is not None: plt.xlim(xlim)
        if ylim is not None: plt.ylim(ylim)
        plt.grid(True, which='both', ls=':')
        plt.legend(fontsize=fs_legend)
        figura+=1
    
        # Fase
        plt.figure(fig_inicial + figura)
        plt.title('Respuesta de Fase', fontsize=fs_title)
        plt.xlabel('Frecuencia [Hz]', fontsize=fs_label)
        plt.ylabel('Fase [°]', fontsize=fs_label)
        plt.tick_params(axis='both', labelsize=fs_ticks)
        
        plt.plot(f, np.degrees(phase), label = f"Filtro {label}")
        if xlim is not None: plt.xlim(xlim)
        plt.grid(True, which='both', ls=':')
        plt.legend(fontsize=fs_legend)
        figura+=1
        
        # Retardo de grupo
        plt.figure(fig_inicial + figura)
        plt.title('Retardo de Grupo', fontsize=fs_title)
        plt.xlabel('Frecuencia [Hz]', fontsize=fs_label)
        plt.ylabel('τg [s]', fontsize=fs_label)
        plt.tick_params(axis='both', labelsize=fs_ticks)
        
        plt.plot(f[:-1], gd, label = f"Filtro {label}")
        if xlim is not None: plt.xlim(xlim)
        plt.grid(True, which='both', ls=':')
        plt.legend(fontsize=fs_legend)
        figura+=1

    # Diagrama de polos y ceros
    if polos_y_ceros == True:
        plt.figure(fig_inicial + figura, figsize = (5,5))
        plt.title(f'Diagrama de Polos y Ceros de {label} (Plano Z)', fontsize=fs_title)
        plt.xlabel('Eje Real', fontsize=fs_label)
        plt.ylabel('Eje Imaginario', fontsize=fs_label)
        plt.tick_params(axis='both', labelsize=fs_ticks)
        
        plt.plot(np.real(p), np.imag(p), 'x', markersize=10, label= f'Polos de {label}')
        if len(z) > 0:
            plt.plot(np.real(z), np.imag(z), 'o', markersize=10, fillstyle='none', label=f'Ceros de {label}')
        plt.axhline(0, color='k', lw=0.5)
        plt.axvline(0, color='k', lw=0.5)
        
        # Grafico el circulo unitario
        unit_circle = patches.Circle((0, 0), radius=1, fill=False, color='gray', ls='dotted', lw=2)
        axes_hdl = plt.gca()
        axes_hdl.add_patch(unit_circle)

        plt.axis([-1.1, 1.1, -1.1, 1.1])
        plt.legend(fontsize=fs_legend)
        plt.grid(True)

def plotear_funcion_transferencia_sos(sos, fs, xlim = None, ylim = None, label = 'T(s)', polos_y_ceros = True, magnitud_fase_retardo = True, fig_inicial = 0, f_banda = None, fs_title=20, fs_label=18, fs_legend=14, fs_ticks=14):
    """
        Plotea la respuesta en frecuencia de una función transferencia de:
            - Módulo
            - Fase
            - Retardo de grupo
            - Polos y Ceros

        b: Vector de coeficientes que modifican la entrada del sistema
        a: Vector de coeficientes que modifican la salida del sistema
        xlim: Limites horizontales del gráfico
        xlim_s: Status del xlim (True si quiero limitar el gráfico, 
                                 False si quiero imprimirlo sin limites)
        fig_inicial: Número que quiero que tenga la primer figura a 
                    imprimir. Sirve para sobre-escribir (o no) plotteos 
                    de diferentes funciones en un mismo gráfico (dejar en 
                    0 si se quiere sobre-plotear)
        f_banda: Tupla con los límites de la banda a sombrear (ej: (8.0, 12.5))
        fs_title: Tamaño de la fuente para los títulos.
        fs_label: Tamaño de la fuente para las etiquetas de los ejes.
        fs_legend: Tamaño de la fuente para las leyendas.
        fs_ticks: Tamaño de la fuente para los números de los ejes.
    """

    # --- Módulo, Fase y Retardo de Fase ---
    f, h = sig.freqz_sos(sos, worN=8192, fs = fs)
    phase = np.unwrap(np.angle(h)) # fase del grupo
    gd = -np.diff(phase) / (2 * np.pi * np.diff(f))
    
    # --- Ploteo ---
    figura = 1
        # Magnitud
    plt.figure(fig_inicial + figura)
    plt.title('Respuesta en Magnitud', fontsize=fs_title)
    plt.xlabel('Frecuencia [Hz]', fontsize=fs_label)
    plt.ylabel('|H(f)| [dB]', fontsize=fs_label)

    curva = plt.plot(f, 20*np.log10(abs(h)), label = f"Filtro {label}") 
    color_curva = curva[0].get_color()
    
    if f_banda is not None: 
        plt.axvspan(f_banda[0], f_banda[1], color=color_curva, alpha=0.15, label=f'Banda {label}', zorder=0)
        
        ticks_actuales = plt.gca().get_xticks() 
        nuevos_ticks = list(ticks_actuales) + [f_banda[0], f_banda[1]]
        nuevos_ticks = sorted(list(set(nuevos_ticks))) 
        
        if xlim is not None:
            nuevos_ticks = [t for t in nuevos_ticks if xlim[0] <= t <= xlim[1]]
            
        etiquetas_escalonadas = []
        nivel_abajo = False
        
        for i in range(len(nuevos_ticks)):
            if i > 0 and (nuevos_ticks[i] - nuevos_ticks[i-1]) < 2.0:
                nivel_abajo = not nivel_abajo 
            else:
                nivel_abajo = False
                
            texto = f"{nuevos_ticks[i]:g}" 
            
            if nivel_abajo:
                etiquetas_escalonadas.append(f"\n{texto}")
            else:
                etiquetas_escalonadas.append(texto)
                
        plt.xticks(nuevos_ticks, etiquetas_escalonadas, rotation=0)
            
        plt.tick_params(axis='both', labelsize=fs_ticks)
        
        if xlim is not None: plt.xlim(xlim)
        if ylim is not None: plt.ylim(ylim)
        plt.grid(True, which='both', ls=':')
        plt.legend(fontsize=fs_legend)
        figura+=1
    
        # Fase
        plt.figure(fig_inicial + figura)
        plt.title('Respuesta de Fase', fontsize=fs_title)
        plt.xlabel('Frecuencia [Hz]', fontsize=fs_label)
        plt.ylabel('Fase [°]', fontsize=fs_label)
        plt.tick_params(axis='both', labelsize=fs_ticks)
        
        plt.plot(f, np.degrees(phase), label = f"Filtro {label}")
        if xlim is not None: plt.xlim(xlim)
        plt.grid(True, which='both', ls=':')
        plt.legend(fontsize=fs_legend)
        figura+=1
        
        # Retardo de grupo
        plt.figure(fig_inicial + figura)
        plt.title('Retardo de Grupo', fontsize=fs_title)
        plt.xlabel('Frecuencia [Hz]', fontsize=fs_label)
        plt.ylabel('τg [s]', fontsize=fs_label)
        plt.tick_params(axis='both', labelsize=fs_ticks)
        
        plt.plot(f[:-1], gd, label = f"Filtro {label}")
        if xlim is not None: plt.xlim(xlim)
        plt.grid(True, which='both', ls=':')
        plt.legend(fontsize=fs_legend)
        figura+=1

def ancho_de_banda(f,Px,porcentaje):
    """
    Descripción: Determina el ancho de banda de una señal a partir de un porcentaje de energía definido.
    """
    pot_acum = np.cumsum(Px) #Potencia acumulada
    pot_acum_norm = pot_acum/pot_acum[-1] # Normalizo
    cond_aux = pot_acum_norm>=(porcentaje/100) # Máscara booleana
    f_umbral = f[cond_aux] # Filtro frecuencias según máscara 
    # idx = np.where(pot_acum_norm >= (porcentaje/100))[0] # Alternativa con índices que devuelve tupla y selecciono la primera en el vector de frecuencias
    f_wb=np.round(f_umbral[0]) # Tomo el primer valor redondeado
    print(f'El ancho de banda {f_wb} Hz contiene el {porcentaje}% de la potencia\n')
    return f_wb

def filtrar_bandas(b, a, dict_crudo, nombre_matriz_filtrada,sujetos, bandas):
    """
    - sujetos: lista de sujetos (sanos o con esquizofrenia)
    - dict_crudo: diccionario con los datos de los sujetos
    - nombre_matriz_filtrado: nombre de la matriz filtrada resultante
    - bandas: lista con los nombres de las bandas a filtrar
    """
    for sujeto in sujetos:
        nombre_matriz_filtrada[sujeto] = {}
        
        # Toma el diccionario y devuelve una matriz de 19 canales x N muestras
        matriz_cruda = dict_crudo[sujeto].get_data() 
        
        for j in range(len(bandas)):
            nombre_banda = bandas[j]

            # Filtramos los 19 canales en el sentido del tiempo
            matriz_filtrada = sig.filtfilt(b[j], a[j], matriz_cruda, axis=1)
            nombre_matriz_filtrada[sujeto][nombre_banda] = matriz_filtrada
        
    return nombre_matriz_filtrada

def aplicar_filtros_ab(lista_sujetos, dict_crudo, b, a, nombres_bandas):
    """
    Descripción: Toma los datos crudos de MNE de varios sujetos, les aplica un 
    banco de filtros IIR en paralelo (axis=1) y devuelve un diccionario anidado.
    
    Entradas:
    - lista_sujetos: Array con los IDs (ej: sujetos_h o sujetos_s)
    - dict_crudo: Diccionario con los objetos RAW de MNE (ej: sujetos_sanos)
    - sos: filtro
    - nombres_bandas: Lista con los nombres de las bandas
    
    Salida:
    - dict_filtrado: Diccionario con estructura [sujeto][banda] = matriz_numpy
    """
    dict_filtrado = {}
    
    for sujeto in lista_sujetos:
        dict_filtrado[sujeto] = {}
        
        # Extraemos la matriz completa de 19 canales para este sujeto
        matriz_cruda = dict_crudo[sujeto].get_data() 
        
        for j in range(len(nombres_bandas)):
            nombre_banda = nombres_bandas[j]

            matriz_filtrada = sig.filtfilt(b = b[j], a = a[j], x = matriz_cruda, axis=1)
            
            dict_filtrado[sujeto][nombre_banda] = matriz_filtrada
            
    return dict_filtrado

def aplicar_filtros_sos(lista_sujetos, dict_crudo, sos, nombres_bandas):
    """
    Descripción: Toma los datos crudos de MNE de varios sujetos, les aplica un 
    banco de filtros IIR en paralelo (axis=1) y devuelve un diccionario anidado.
    
    Entradas:
    - lista_sujetos: Array con los IDs (ej: sujetos_h o sujetos_s)
    - dict_crudo: Diccionario con los objetos RAW de MNE (ej: sujetos_sanos)
    - sos: filtro
    - nombres_bandas: Lista con los nombres de las bandas
    
    Salida:
    - dict_filtrado: Diccionario con estructura [sujeto][banda] = matriz_numpy
    """
    dict_filtrado = {}
    
    for sujeto in lista_sujetos:
        dict_filtrado[sujeto] = {}
        
        # Extraemos la matriz completa de 19 canales para este sujeto
        matriz_cruda = dict_crudo[sujeto].get_data() 
        
        for j in range(len(nombres_bandas)):
            nombre_banda = nombres_bandas[j]

            matriz_filtrada = sig.sosfiltfilt(sos = sos[j], x = matriz_cruda, axis=1)
            
            dict_filtrado[sujeto][nombre_banda] = matriz_filtrada
            
    return dict_filtrado

def matriz_conectividad_PLI(lista_sujetos, dict_filtrado, banda, fmin, fmax, fs=250, segundos_epoca=2):
    """
    Descripción: Calcula las matrices de conectividad simétricas (PLI) para un grupo de sujetos.
    
    Entradas:
    - lista_sujetos: Array con los IDs de los sujetos (ej: sujetos_h)
    - dict_filtrado: Diccionario anidado con los datos ya filtrados
    - banda: String con el nombre de la banda (ej: 'Alpha')
    - fmin, fmax: Límites de frecuencia para la extracción de fase
    - fs: Frecuencia de muestreo (por defecto 250 Hz)
    - segundos_epoca: Tamaño de la ventana de tiempo (por defecto 2 segundos)
    
    Salida:
    - matrices_pli: Lista conteniendo las matrices de adyacencia (19x19) de cada sujeto
    """
    muestras_epoca = segundos_epoca * fs
    matrices_pli = []
    
    for sujeto in lista_sujetos:
        # Extraemos la señal del sujeto
        datos_filtrados = dict_filtrado[sujeto][banda]
        
        # Recortamos en épocas
        n_epocas = datos_filtrados.shape[1] // muestras_epoca
        datos_recortados = datos_filtrados[:, :n_epocas * muestras_epoca]
        datos_3d = datos_recortados.reshape(19, n_epocas, muestras_epoca).transpose(1, 0, 2)
        
        # Calculamos PLI
        resultado = spectral_connectivity_epochs(
            data=datos_3d, method='pli', sfreq=fs, 
            fmin=fmin, fmax=fmax, faverage=True, verbose=False
        )
        
        # Espejamos la matriz (triángulo superior + transpuesta) y guardamos
        matriz_mne = resultado.get_data(output='dense')[:, :, 0]
        matriz_simetrica = matriz_mne + matriz_mne.T
        matrices_pli.append(matriz_simetrica)
        
    return matrices_pli

def limpiar_artefactos_ica(lista_sujetos, dict_crudo, fs):
    """
    Descripción: Aplica ICA a las señales continuas para detectar y eliminar 
    automáticamente los artefactos de parpadeo usando Fp1 como referencia.
    """
    dict_limpio = {}
    print("\nIniciando extracción de artefactos con ICA...")
    
    # fs=250
    # cant_coef_hp = 5601
    
    # ripple = 0.1   # dB
    # attenuation = 25  # dB
    
    # fstop_hp = 0.7  # Hz
    # fpass_hp = 1  # Hz
    
    # frecs_hp = [0.0, fstop_hp, fpass_hp, fs/2]
    # gains_db_hp = [-np.inf, -attenuation, -ripple, 0]
    # gains_hp = 10**(np.array(gains_db_hp)/20)
    
    # #num_hp = sig.firwin2(cant_coef_hp, frecs_hp, gains_hp, window=('kaiser', 14), fs=fs)
    
    # num_hp = sig.firls(numtaps = cant_coef_hp, bands = frecs_hp, desired = gains_hp, fs = fs)
    
    # # Visualización del pasa altos
    # w_rad = 2 * np.pi * np.linspace(0.1, fs/2, 1000) / fs
    # w, hh_hp = sig.freqz(num_hp, worN=w_rad)
    
    # plt.figure()
    # plt.plot(w * fs / (2 * np.pi), 20 * np.log10(np.abs(hh_hp) + 1e-15), label='Pasa altos')
    # plot_plantilla(filter_type='highpass', fpass=fpass_hp, ripple=1, fstop=fstop_hp, attenuation=attenuation, fs=fs)
    # plt.title('Filtro pasa altos', fontsize=18)
    # plt.xlabel('Frecuencia [Hz]', fontsize=16)
    # plt.ylabel('Magnitud [dB]', fontsize=16)
    # plt.tick_params(axis='both', labelsize=14)
    # plt.grid(which='both', axis='both')
    # plt.legend(fontsize=14)
    # plt.show()    
    
    
    sos = sig.iirdesign(wp = 1.1, ws = 0.7, gpass = 0.5, gstop = 30, 
                        analog = False, ftype = 'butter', output = 'sos', fs = fs)

    f, h = sig.sosfreqz(sos, worN=4000, fs=fs)

    magnitud_db = 20 * np.log10(np.abs(h) + 1e-15)
    
    # --- Configuración del Ploteo ---
    plt.figure(figsize=(10, 6))    
    plt.plot(f, magnitud_db, label='Filtro pasa altos')
    plt.axhline(-3, color='red', ls='--', alpha=0.5, label='Atenuación de la frecuencia de corte (ω0)')
    plt.axhline(-30, color='red', ls='--', alpha=0.5, label='Atenuación mínima (gstop = -30 dB)')
    plt.title('Respuesta en Magnitud', fontsize=20)
    plt.xlabel('Frecuencia [Hz]', fontsize=18)
    plt.ylabel('|H(f)| [dB]', fontsize=18)
    plt.xlim([0, 2]) 
    plt.ylim([-40, 1])
    ticks_base = list(np.arange(-40, 1, 10))
    nuevos_ticks_y = sorted(ticks_base + [-3.0]) 
    plt.yticks(nuevos_ticks_y) 
    plt.tick_params(axis='both', labelsize=14)
    plt.grid(True, which='both', ls=':')
    plt.legend(fontsize=14)
    plt.show()
    
    for sujeto in lista_sujetos:
        raw_sujeto = dict_crudo[sujeto].copy()
         
        matriz_cruda = raw_sujeto.get_data()
        
        matriz_filtrada = sig.sosfiltfilt(sos, matriz_cruda, axis=1) #1 porque es un filt
        
        #matriz_filtrada = sig.filtfilt(b, a, matriz_cruda, axis=1)
        
        info_actualizada = raw_sujeto.info.copy()
        
        try:
            info_actualizada['highpass'] = 1
        except RuntimeError:
            with info_actualizada._unlock():
                info_actualizada['highpass'] = 1
        
        raw_para_ica = mne.io.RawArray(matriz_filtrada, info_actualizada, verbose=False)
        
        # 3. Configuramos el motor matemático ICA
        # Le pedimos que encuentre 15 componentes independientes (voces)
        ica = ICA(n_components=15, method='fastica', random_state=42, max_iter='auto')
        ica.fit(raw_para_ica, verbose=False)
        
        # 4. Búsqueda automática del parpadeo (Usamos Fp1 como "buchón" o EOG virtual)
        indices_malos, scores = ica.find_bads_eog(raw_sujeto, ch_name='Fp1', verbose=False)
        
        # Le decimos a ICA cuáles son los componentes que hay que borrar
        ica.exclude = indices_malos
        
        # 5. Aplicamos la limpieza a nuestra señal original
        raw_limpio = raw_sujeto.copy()
        ica.apply(raw_limpio, verbose=False)
        
        # Guardamos el raw limpio en nuestro nuevo diccionario
        dict_limpio[sujeto] = raw_limpio
        print(f"[{sujeto}] ICA completado: Se borraron {len(indices_malos)} componentes de parpadeo.")
        
    return dict_limpio