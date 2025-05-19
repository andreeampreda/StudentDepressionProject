/* 1. Incarcarea setului de date */
data depression_data;
    infile '/home/u64202338/Student Depression Dataset.csv' dsd dlm=',' firstobs=2;
    input id
          Gender :$10.
          Age
          City :$30.
          Profession :$15.
          Academic_Pressure
          Work_Pressure
          CGPA
          Study_Satisfaction
          Job_Satisfaction
          Sleep_Duration :$20.
          Dietary_Habits :$15.
          Degree :$15.
          Suicidal_Thoughts :$35.
          Work_Study_Hours
          Financial_Stress
          Mental_Illness_History :$30.
          Depression;
run;

/* Eliminarea coloanelor */
data depression_data;
    set depression_data;
    drop Profession Work_Pressure Job_Satisfaction;
run;


/* 2. Definirea formatelor */
proc format;
    value gender_fmt       0 = 'Male' 1 = 'Female';
    value thoughts_fmt 	   0 = 'No'  1 = 'Yes';
    value diet_fmt
       0= 'Unhealthy'
       1= 'Moderate' 
       2= 'Healthy' 
       9= 'Others';
    value sleep_fmt
        0= 'Less than 5 hours' 
        1= '5-6 hours'         
        2= '7-8 hours'         
        3= 'More than 8 hours' 
         9= 'Others';
   
    value mental_fmt 1= 'Yes'  0= 'No';
run;

/* 3. Frecventa depresiei in functie de gen */

title "Frecventa depresiei in functie de gen";
proc freq data=depression_data_formatted;
    tables Gender_num * Depression;
    format Gender_num gender_fmt.
           Depression depression_fmt.;
run;


/* 4. Interogare SQL asupra setului de date */
proc sql;
    select Gender_num format=gender_fmt., 
           count(*) as Numar_Studenti
    from depression_data_formatted
    group by Gender_num;
quit;




/* 5. Statistici descriptive pentru stres si performanta */
proc means data=depression_data mean std min max maxdec=2;
    var Academic_Pressure Financial_Stress CGPA;
run;


/* 6. Grupare CGPA in categorii  */
data depression_data;
    set depression_data;
    if CGPA < 2 then CGPA_Level = "Scazut";
    else if 2 <= CGPA < 3 then CGPA_Level = "Mediu";
    else if CGPA >= 3 then CGPA_Level = "Ridicat";
run;

proc freq data=depression_data;
    tables CGPA_Level;
run;

/* 7. Corelatie intre stres si depresie */
proc corr data=depression_data;
    var Academic_Pressure Financial_Stress Work_Study_Hours;
    with Depression;
run;

/* 8. Grafic: depresie in functie de durata somnului */
proc gchart data=depression_data;
    vbar Sleep_Duration / subgroup=Depression;
run;

