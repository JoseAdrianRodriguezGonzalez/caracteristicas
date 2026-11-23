import matplotlib.pyplot as plt
import numpy as np 
#from mlp_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from PIL import Image
def kmeans_plots(dictio,save_path):
    cohesion=dictio["cohesion"]
    intra_extra=dictio["all_distances"]
    colors=['blue','red']
    _,(ax1,ax2)=plt.subplots(1,2,figsize=(12,5))
    for label in cohesion:
        n=cohesion[label]["all_distances"]
        mask=np.isin(intra_extra[:,label],n)
        cluster_distances=intra_extra[mask]
        intra=cluster_distances[:,label]
        inter=cluster_distances[:,1-label]
        jitter=np.random.normal(0,0.1,len(intra))
        x_pos=label+1+jitter
        ax1.scatter(intra,x_pos, alpha=0.4, color=colors[label], 
        label=f'Cluster {label}', s=30)
        
        # Subplot 2: Distancias inter-cluster
        ax2.scatter(inter,x_pos, alpha=0.4, color=colors[label], 
        label=f'Cluster {label}', s=30) 
    # Configurar subplot 1: Intra-cluster
    ax1.set_xlabel('Distancia Intra-Cluster')
    ax1.set_ylabel('Cluster')
    ax1.set_yticks([1, 2])
    ax1.set_yticklabels(['Cluster 0', 'Cluster 1'])
    ax1.set_title('Distancias Intra-Cluster')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Configurar subplot 2: Inter-cluster  
    ax2.set_xlabel('Distancia Inter-Cluster')
    ax2.set_ylabel('Cluster')
    ax2.set_yticks([1, 2])
    ax2.set_yticklabels(['Cluster 0', 'Cluster 1'])
    ax2.set_title('Distancias Inter-Cluster')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/kmeans_distances.png",dpi=300)
    #plt.show()        
def QD(dictio,save_path):
    quality=dictio["confianza"]
    plt.figure(figsize=(12,5))
    plt.hist(quality,bins=20,alpha=0.7,color="skyblue",edgecolor="black")
    plt.xlabel("Valor de confianza")
    plt.ylabel("Frecuencia")
    plt.title("Distribucion de la Confianza del Clustering")
    plt.grid(True,alpha=0.3)
    plt.savefig(f"{save_path}/confianza_histograma.png", dpi=300)
    #plt.show()
def plot_frame_3d(dataset,frame_index=0,save_path=""):
    frame_data=dataset.iloc[frame_index].values
    points=frame_data.reshape(-1,3)
    connections = [
        (3,2),(2,20),(20,1),(1,0),(0,12),(16,17),(12,13),(17,18),(18,19),(13,14),(14,15),
        (0,16),(20,4),(20,8),(8,9),(9,10),(10,11),(4,5),(5,6),(6,7),(7,21),(11,23)
    ]
    fig=plt.figure(figsize=(10,8))
    ax=fig.add_subplot(111,projection='3d')
    x=points[:,0]
    y=points[:,1]
    z=points[:,2]
    scatter=ax.scatter(x,y,z,c=range(25),cmap='viridis',s=50,alpha=0.7)
    for start, end in connections:
        if start < len(points) and end < len(points):
            ax.plot([x[start], x[end]], [y[start], y[end]], [z[start], z[end]], 
                    'red', alpha=0.7, linewidth=2)    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Frame {frame_index} - 25 Puntos 3D')
    plt.colorbar(scatter, ax=ax, label='Índice del Punto')
    for i in range(25):
        ax.text(x[i], y[i], z[i], f'{i}', fontsize=8) 
    plt.savefig(f"{save_path}/frame_{frame_index}.png", dpi=300)
    #plt.show()
    
    return points
def plot_multiple_frames(dataset,start_frame=0,end_frame=10,frames_per_plot=10,save_path=""):
    total_frames=end_frame-start_frame+1 
    num_groups=total_frames//frames_per_plot 
    groups=[]
    for i in range(num_groups):
        group_start=start_frame +(i*frames_per_plot)
        groupd_end=group_start +frames_per_plot-1
        group_frames=list(range(group_start,groupd_end+1))
        groups.append(group_frames)

    for group_idx, frame_indice in enumerate(groups):
        n_frames=len(frame_indice)
        fig=plt.figure(figsize=(5*n_frames,4))
        fig.suptitle(f'Grupo {group_idx + 1}: Frames {frame_indice[0]} a {frame_indice[-1]}', 
                    fontsize=16, fontweight='bold')
        for i, frame_idx in enumerate(frame_indice):
            frame_data=dataset.iloc[frame_idx].values

            points=frame_data.reshape(-1,3)
            ax=fig.add_subplot(1,n_frames,i+1,projection='3d')
            x,y,z=points[:,0],points[:,1],points[:,2]
            connections = [
                (3,2),(2,20),(20,1),(1,0),(0,12),(16,17),(12,13),(17,18),(18,19),(13,14),(14,15),
                (0,16),(20,4),(20,8),(8,9),(9,10),(10,11),(4,5),(5,6),(6,7),(7,21),(11,23)
            ]
            ax.scatter(x,y,z,c=range(25),cmap='viridis',s=50,alpha=0.7)
            for start, end in connections:
                if start < len(points) and end < len(points):
                    ax.plot([x[start], x[end]], [y[start], y[end]], [z[start], z[end]], 
                            'red', alpha=0.7, linewidth=2)    
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Frame {frame_idx} - 25 Puntos 3D')
        plt.tight_layout()
        plt.savefig(f"{save_path}/grupo_{group_idx+1}_frames_{frame_indice[0]}_{frame_indice[-1]}.png", dpi=300)
        #plt.show()
def plot_consecutive_frames_same_plot(dataset, start_frame=0, end_frame=100, frames_per_group=10,save_path=""):
    total_frames = end_frame - start_frame + 1
    num_groups = total_frames // frames_per_group
    for group_idx in range(num_groups):
        group_start = start_frame + (group_idx * frames_per_group)
        group_end = group_start + frames_per_group - 1
        frame_indices = list(range(group_start, group_end + 1))
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        connections = [
            (3,2),(2,20),(20,1),(1,0),(0,12),(16,17),(12,13),(17,18),(18,19),(13,14),(14,15),
            (0,16),(20,4),(20,8),(8,9),(9,10),(10,11),(4,5),(5,6),(6,7),(7,21),(11,23)
        ]
        for i, frame_idx in enumerate(frame_indices):
            frame_data = dataset.iloc[frame_idx].values
            points = frame_data.reshape(-1, 3)
            x, y, z = points[:, 0], points[:, 1], points[:, 2]
            color = colors[i % len(colors)]   
            ax.scatter(x, y, z, c=color, s=30, alpha=0.7, label=f'Frame {frame_idx}')
            for start, end in connections:
                if start < len(points) and end < len(points):
                    ax.plot([x[start], x[end]], [y[start], y[end]], [z[start], z[end]], 
                            color=color, alpha=0.3, linewidth=1)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Grupo {group_idx + 1}: Frames {group_start} a {group_end}')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{save_path}/grupo_{group_idx+1}_frames_{group_start}_{group_end}.png", dpi=300)
     #   plt.show()
        print(f"Grupo {group_idx + 1}: Frames {group_start} a {group_end}")
def plot_gif(dataset,start_frame=0,num_frames=5,output_path=""):
    fig=plt.figure(figsize=(10,8))
    ax=fig.add_subplot(111,projection='3d')
    connections = [
        (3,2),(2,20),(20,1),(1,0),(0,12),(16,17),(12,13),(17,18),(18,19),(13,14),(14,15),
        (0,16),(20,4),(20,8),(8,9),(9,10),(10,11),(4,5),(5,6),(6,7),(7,21),(11,23)
    ]
    def update(frame):
        ax.clear()
        frame_idx=start_frame+frame
        frame_data=dataset.iloc[frame_idx].values 
        points=frame_data.reshape(-1,3)
        x,y,z=points[:,0],points[:,1],points[:,2]
        scatter=ax.scatter(x,y,z,c=range(25),cmap='viridis',s=50,alpha=0.7)
        for start, end in connections:
            if start < len(points) and end < len(points):
                ax.plot([x[start], x[end]], [y[start], y[end]], [z[start], z[end]], 
                        'red', alpha=0.7, linewidth=2)    
        ax.set_title(f'frame {frame_idx}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim([np.min(dataset.values[:, 0::3]), np.max(dataset.values[:, 0::3])])
        ax.set_ylim([np.min(dataset.values[:, 1::3]), np.max(dataset.values[:, 1::3])])
        ax.set_zlim([np.min(dataset.values[:, 2::3]), np.max(dataset.values[:, 2::3])])
        return scatter 
    anim=FuncAnimation(fig,update,frames=num_frames,interval=100,blit=False)
    anim.save(output_path,writer="pillow",fps=10)
    plt.close() 
    print("fgif guardado en {output_path}")
def plot_group_gifs(dataset,start_frame=0,end_frame=100,frames_per_group=10):
    total_frames=end_frame-start_frame+1
    print(total_frames)
    num_groups=total_frames//frames_per_group
    for i in range(num_groups):
        group_start=start_frame+(i*frames_per_group)
        group_end=group_start+frames_per_group-1
        output_path = f"grupo_{i+1}_frames_{group_start}_{group_end}.gif"
        print(output_path)
        # Usar la función plot_gif existente para cada grupo
        plot_gif(dataset,group_start, frames_per_group, output_path=output_path)
