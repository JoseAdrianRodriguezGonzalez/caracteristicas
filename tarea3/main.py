from scipy.sparse import data
from src.pipelines.apply_filter import Filters 
from src.pipelines.distance import distance
from src.pipelines.features import Features
from utils.save_files import save_file
from entrypoint.read import read_output
from model.kmeans import kmeans_metrics
from utils.save_logs import save_log
from sklearn.preprocessing import MinMaxScaler
from utils.save_plots import kmeans_plots,QD ,plot_frame_3d,plot_gif,plot_multiple_frames,plot_consecutive_frames_same_plot,plot_group_gifs
import os 
def build_dataset():
    f=Filters(5,"data/Joints_Data.txt")
    df=f.filtro_medio()
    n_frames =len(df)-10
    step = 10
    os.makedirs("assets",exist_ok=True)
    for start in range(8,n_frames,step):
        end=min(start+step-1,n_frames-1)
        print(f"Procesando frames de {start} a {end}")

        plot_frame_3d(df.iloc[start:end,1:],save_path="assets")
        plot_multiple_frames(df.iloc[:,1:],start,end,step,save_path="assets")
        plot_consecutive_frames_same_plot(df.iloc[:,1:],start,end,step,save_path="assets")
    plot_group_gifs(df.iloc[:,1:],0,n_frames,step)
    
    # graficar de a 10 frames
    #df=f.data 
    #joints=[3,5,7,9,11,13,15,17,19,4,8,12,16]
    joints=[7,11,15,19,4,8,12,16]
    D=distance(df,joints,0)
    Complete_distance=D.calculate_distance()
    F=Features(Complete_distance,loss_start=0,loss_end=0)
    features_complete=F.features_all()
    save_file(features_complete,"features.csv")

def main():
    build_dataset()
    dataframe=read_output("out/features.csv")
    print(dataframe.info())
    
    X=dataframe.iloc[1:,:]
    scaler=MinMaxScaler()
    X_scaled=scaler.fit_transform(X)
    metricas=kmeans_metrics(X)

    save_log(metricas,"log.txt")
    kmeans_plots(metricas,save_path="assets")
    QD(metricas,save_path="assets")
if __name__ =="__main__":
    main()
