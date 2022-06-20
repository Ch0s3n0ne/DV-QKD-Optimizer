import time
# %matplotlib qt5
import numpy as np
import nidaqmx
from nidaqmx.constants import (
TerminalConfiguration)
import matplotlib.pyplot as plt


def f(measured_array,goal_array,current_volt,last_volt,weight_1,weight_2,n_stages):
    
    first_sum=0
    second_sum=0
    
    for i in range(0,3):
        #first_sum+=abs(current_array[i]-goal_array[i])
        first_sum+=abs(measured_array[i]-goal_array[i])*weight_1
        # print(abs(measured_array[i]-goal_array[i]))
        
    # print("////")
    
    for i in range(0,n_stages*2):
        #first_sum+=abs(current_array[i]-goal_array[i])
        second_sum+=abs(current_volt[i]-last_volt[i])*weight_2
        
        # print(abs(current_volt[i]-last_volt[i]))
         
    z=(first_sum/3)+(second_sum/(n_stages*2))
    
    #print(first_sum,'~~~',second_sum,'~~',third_sum,'~~>',z)
    return z

def make_measurement():
    ''' 
    Setup:
    (Differential mode)
    S1 -> Dev1/ai0
    S2 -> Dev1/ai1
    S3 -> Dev1/ai2
    DOP or Power -> Dev1/ai3

    Ground connected to AI4,AI5,AI6,AI7

    (Rising Edge Mode)
    Trigger output -> Dev1/port0/line0 

    '''
    
    r_analog_input=["Dev1/ai0","Dev1/ai1","Dev1/ai2","Dev1/ai3"]
    o_trigger="Dev1/port0/line0"
    
    task_1=nidaqmx.Task() 
    task_2=nidaqmx.Task() 
    task_3=nidaqmx.Task() 
    task_4=nidaqmx.Task()
    task_5=nidaqmx.Task()

    task_1.do_channels.add_do_chan("Dev1/port0/line0")
    task_2.ai_channels.add_ai_voltage_chan(r_analog_input[0],terminal_config=TerminalConfiguration.RSE)
    task_3.ai_channels.add_ai_voltage_chan(r_analog_input[1],terminal_config=TerminalConfiguration.RSE)
    task_4.ai_channels.add_ai_voltage_chan(r_analog_input[2],terminal_config=TerminalConfiguration.RSE)
    task_5.ai_channels.add_ai_voltage_chan(r_analog_input[3],terminal_config=TerminalConfiguration.RSE)
    
    # task_1.write(True)
    
    # time.sleep(0.25)
    
    r_multiple_read=np.zeros((3,8))
    r_measured_values=np.zeros(3)
    
      
    r_multiple_read[0]=task_2.read(number_of_samples_per_channel=8)
    r_multiple_read[1]=task_3.read(number_of_samples_per_channel=8)
    r_multiple_read[2]=task_4.read(number_of_samples_per_channel=8)
    #r_multiple_read[3]=task_5.read(number_of_samples_per_channel=8)

    r_measured_values[0]=np.mean(r_multiple_read[0,:])
    r_measured_values[1]=np.mean(r_multiple_read[1,:])
    r_measured_values[2]=np.mean(r_multiple_read[2,:])
    #r_measured_values[3]=np.mean(r_multiple_read[3,:])


    # task_1.write(False)   
    
    r_output_values=r_measured_values/2.5
    
    task_1.stop()
    task_1.close()
    task_2.stop()
    task_2.close()
    task_3.stop()
    task_3.close()
    task_4.stop()
    task_4.close()
    task_5.stop()
    task_5.close()
    
    return(r_output_values)
    
    


n_stages=6
n_particles=6
voltage_range=10 #deve ser necessário 20

goal=np.array([1,0,0])

for i in range(0,3):
    if(goal[i]==1):
        pol_index=i
        signal=1
    if(goal[i]==-1):
        pol_index=i
        signal=-1

# print(pol_index,signal)

''' De momento vou considerar em grande parte irrelevante o efeito de V0  e Vpi  
    visto que o maior aumento nas transições devido a eles seria apenas de 
    cerca de 3V
'''
V_A_array=np.array([-10.35198091 , -9.11390144  ,-7.22656287 ,-10.3402543 ,  -7.32236055
  ,-7.86467429] )
V_C_array=np.array([ 6.13810597 , 5.11179801 , 3.02560257, 7.31351766 ,11.19369122 ,12.28584249])

V_A_bias_array=np.array([-10.7,-9.3,-8.5,-10.9,-7.4,-7.6])
V_C_bias_array=np.array([8.4,9.6,9.4,11.1,11.6,10.9])

V_bias_array=np.concatenate((V_A_bias_array, V_C_bias_array))

V_A_array_m_bias=V_A_array-V_A_bias_array
V_C_array_m_bias=V_C_array-V_C_bias_array

V_A_bias_array_up=np.zeros(n_stages)
V_A_bias_array_low=np.zeros(n_stages)

V_C_bias_array_up=np.zeros(n_stages)
V_C_bias_array_low=np.zeros(n_stages)

for i in range(0,n_stages):
     
    V_A_bias_array_up[i]=V_A_bias_array[i]+voltage_range/2
    V_A_bias_array_low[i]=V_A_bias_array[i]-voltage_range/2
    
    V_C_bias_array_up[i]=V_C_bias_array[i]+voltage_range/2
    V_C_bias_array_low[i]=V_C_bias_array[i]-voltage_range/2
    
V_A_bias_mod=np.zeros((n_stages,n_particles))
V_C_bias_mod=np.zeros((n_stages,n_particles))

for j in range(0,n_particles):
    
    for i in range(0,n_stages):
        value_A=np.random.rand(1)*(V_A_bias_array_up[i]-V_A_bias_array_low[i])+V_A_bias_array_low[i]
        value_C=np.random.rand(1)*(V_C_bias_array_up[i]-V_C_bias_array_low[i])+V_C_bias_array_low[i]
        
        V_A_bias_mod[i,j]=value_A[0]
        V_C_bias_mod[i,j]=value_C[0]

X = np.concatenate((V_A_bias_mod, V_C_bias_mod))
V = np.random.randn(n_stages*2, n_particles) * 0.1 #distribuição normal

run=1
run_part=1
next_run=0

while(run):
    
    pbest=X
    #aplicamos os valores das coordadenadas de cada partícula à função
    pbest_obj=np.zeros(n_particles)

    
    file = open("record.txt", "w")
    str_pbest=repr(pbest)
    str_pbest_obj=repr(pbest_obj)
    file.write("pbest = " + str_pbest + "\n")
    file.write("#\n")
    file.write("pbest_obj = " + str_pbest_obj + "\n")
    file.write("@\n")
    file.close()

    
    particle_number=1
    
    while(particle_number<=n_particles and run_part):
        
        i=particle_number-1
        print("Partícula número:",particle_number)
        print("Valores de tensão \nVA1",V_A_array_m_bias[0]+X[0,i],"\nVC1",V_C_array_m_bias[0]+X[6,i],"\nVA2",V_A_array_m_bias[1]+X[1,i],"\nVC2",V_C_array_m_bias[1]+X[7,i],"\nVA3",V_A_array_m_bias[2]+X[2,i],"\nVC3",V_C_array_m_bias[2]+X[8,i],
            "\nVA4",V_A_array_m_bias[3]+X[3,i],"\nVC4",V_C_array_m_bias[3]+X[9,i],"\nVA5",V_A_array_m_bias[4]+X[4,i],"\nVC5",V_C_array_m_bias[4]+X[10,i],"\nVA6",V_A_array_m_bias[5]+X[5,i],"\nVC6",V_C_array_m_bias[5]+X[11,i])
        
        user_input = input("\nMake measurement, type stop to end:\n")
        
        if user_input=="stop":
            run_part=0
            run=0

        if(run_part==1):
            
            r_measurment=make_measurement()
            
            print("Os valores medidos para a partícula:",particle_number,"-->",r_measurment,"\n")

            max_value=np.sqrt(r_measurment[0]**2+r_measurment[1]**2+r_measurment[2]**2)
            
            lab_goal=np.zeros(3)
            
            lab_goal[pol_index]=max_value*signal
            
            pbest_obj[particle_number-1]=f(r_measurment,lab_goal,X[:,i],V_bias_array,5,0.5,n_stages)
            
            gbest = pbest[:, pbest_obj.argmin()]
            #coloca o valor que a função objetivo deu para essas coordenadas 
            gbest_obj = pbest_obj.min()
            
            if(particle_number==n_particles):
                next_run=1
                
                print("pbest_obj=",pbest_obj)
                print("gbest=",gbest)
                
            particle_number+=1    
    
    file = open("record.txt", "w")
    str_pbest=repr(pbest)
    str_pbest_obj=repr(pbest_obj)
    file.write("pbest = " + str_pbest + "\n")
    file.write("#\n")
    file.write("pbest_obj = " + str_pbest_obj + "\n")
    file.write("@\n")
    file.close()
            
    iteration=0
    while(next_run):
        
        c1 = c2 = 0.2
        #incial era 0.8
        w = 0.6
        #fator aleatório
        r = np.random.rand(2)
        #atualização das partículas
        #substrainos as duas matrizes --- subtraimos a coluna de melhores valores a toda a coluna de X
        V = w * V + c1*r[0]*(pbest - X) + c2*r[1]*(gbest.reshape(-1,1)-X)
        X = X + V

        obj=np.zeros(n_particles)
        
        particle_number=1
        run_part=1
        
        while(particle_number<=n_particles and run_part==1):
            
            print("Iteração número:",iteration)
            print("Partícula número:",particle_number)           
            i=particle_number-1
            print("Valores de tensão \nVA1",V_A_array_m_bias[0]+X[0,i],"\nVC1",V_C_array_m_bias[0]+X[6,i],"\nVA2",V_A_array_m_bias[1]+X[1,i],"\nVC2",V_C_array_m_bias[1]+X[7,i],"\nVA3",V_A_array_m_bias[2]+X[2,i],"\nVC3",V_C_array_m_bias[2]+X[8,i],
            "\nVA4",V_A_array_m_bias[3]+X[3,i],"\nVC4",V_C_array_m_bias[3]+X[9,i],"\nVA5",V_A_array_m_bias[4]+X[4,i],"\nVC5",V_C_array_m_bias[4]+X[10,i],"\nVA6",V_A_array_m_bias[5]+X[5,i],"\nVC6",V_C_array_m_bias[5]+X[11,i])
            
            user_input = input("\nMake measurement, type stop to end:\n ")
            
            if user_input=="stop":
                run_part=0
                next_run=0
                run=0

            if(run_part==1):
                
                r_measurment=make_measurement()
                
                print("Os valores medidos para a partícula:",particle_number,"-->",r_measurment,"\n")
                
                file = open("record.txt", "a")
                str_r_measurment=repr(r_measurment)
                str_particle_number=repr(particle_number)
                file.write("medição partícula "+ str_particle_number +"=" + str_r_measurment + "\n")
                file.write("$\n")
                file.close()

                max_value=np.sqrt(r_measurment[0]**2+r_measurment[1]**2+r_measurment[2]**2)
                
                lab_goal=np.zeros(3)
                
                lab_goal[pol_index]=max_value*signal
                
                obj[particle_number-1]=f(r_measurment,lab_goal,X[:,i],V_bias_array,5,0.5,n_stages)
                
                #if(particle_number==n_particles):
                              
                    
                    
                particle_number+=1    
        
        if(run_part==1): 
        
            pbest[:, (pbest_obj >= obj)] = X[:, (pbest_obj >= obj)]
            #atualizar o melhor valor atingido pela particula
            pbest_obj = np.array([pbest_obj, obj]).min(axis=0)
            gbest = pbest[:, pbest_obj.argmin()]
            gbest_obj = pbest_obj.min()
            
        print("pbest_obj=",pbest_obj)
        print("gbest=",gbest)      
            
        file = open("record.txt", "a")
        str_pbest=repr(pbest)
        str_pbest_obj=repr(pbest_obj)
        file.write("pbest = " + str_pbest + "\n")
        file.write("#\n")
        file.write("pbest_obj = " + str_pbest_obj + "\n")
        file.write("@\n")
        file.close()
                
        iteration+=1
