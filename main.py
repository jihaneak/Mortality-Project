import os
import pandas as pd
from src.io.hmd_reader import read_hmd_csv
from src.models.lee_carter import fit_lee_carter, forecast_kt, reconstruct_mx
from src.models.life_expectancy import compute_life_expectancy
from src.utils.plotting import plot_e0_forecast

def main():
    # 1. Configuration des chemins et paramètres
    # Facilement adaptable pour le Maroc ou d'autres segments (Male/Female)
    deaths_path = "data/france_deaths_clean.csv"
    pop_path = "data/france_population_clean.csv"


    # 2. Chargement des données
    # On filtre en amont pour garantir la qualité de la tendance (1970-2019)
    df_hmd = read_hmd_csv(deaths_path, pop_path)
    df_input = df_hmd[(df_hmd['Sex'] == 'Total') & 
                      (df_hmd['Year'] >= 1970) & 
                      (df_hmd['Year'] <= 2019)].copy()
    
    # Calcul du taux de mortalité brut
    df_input['mx'] = df_input['Deaths'] / df_input['Exposure']

    # 3. Calibration du modèle Lee-Carter
    # Gère en interne le signe de la SVD et la normalisation
    ax, bx, kt = fit_lee_carter(df_input)
    print("Modèle Lee-Carter calibré avec succès.")

    # 4. Projection de kt (Horizon 20 ans : 2020-2039)
    kt_future = forecast_kt(kt, n_years=20)

    # 5. Reconstruction des mx futurs
    # Retourne un DataFrame (Ages x Années Projetées)
    mx_future = reconstruct_mx(ax, bx, kt_future)

    print(f"mx Age 0 (An 1) : {mx_future.iloc[0, 0]}")
    print(f"mx Age 0 (An 20) : {mx_future.iloc[0, -1]}")

    if mx_future.iloc[0, 0] == mx_future.iloc[0, -1]:
       print("ALERTE : Les mx futurs sont encore constants !")

    # 6. Calcul de l'espérance de vie (e0)
    # Gère la conversion mx -> qx et la fermeture de table (somme cumulative)
    e0_future = compute_life_expectancy(mx_future)
    
    # Correction des index pour l'affichage (Années réelles au lieu de 1, 2, 3...)
    last_year = df_input['Year'].max()
    e0_future.index = range(last_year + 1, last_year + 1 + len(e0_future))

    # 7. Affichage des résultats
    print(f"Espérance de vie en {e0_future.index[0]} : {e0_future.iloc[0]:.2f} ans")
    print(f"Espérance de vie en {e0_future.index[-1]} : {e0_future.iloc[-1]:.2f} ans")

    # 8. Sauvegarde et Visualisation
    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)
    
    e0_future.to_csv("outputs/tables/e0_forecast.csv")
    plot_e0_forecast(e0_future, save_path="outputs/figures/e0_forecast.png")

    print("Pipeline terminé avec succès ! Graphique disponible dans outputs/figures/")

if __name__ == "__main__":
    main()