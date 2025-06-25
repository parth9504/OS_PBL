import pyttsx3
import speech_recognition as sr
import random
import webbrowser
import datetime
from plyer import notification
import pyautogui
import wikipedia
import platform
import psutil
import speedtest
import subprocess
from pptx import Presentation

from create_ppt import generate_slide_contents,create_multi_slide_presentation

engine = pyttsx3.init()
#store all the different voices available..
voices=engine.getProperty('voices')
#set the voice
engine.setProperty('voice',voices[0].id)
#set the rate of speech..
engine.setProperty("rate",190)
    


def command():
        
    #initialize the speech recognition instance..
    r=sr.Recognizer()
    content=""
    while content=="":
        with sr.Microphone() as source:
            print("Hey I'm Iris")
            audio=r.listen(source)
            
        #using google speech recognizer..
        try:
            content=r.recognize_google(audio,language='en-in')
            print("Recognising....")
            #recognize the audio (language set to indian english)
            print("User: ",content)
            
        except Exception as e:
            print("Please try again")
    
    return content
    
    
#function to speak the command.
def speak(audio):
    engine.say(audio)
    engine.runAndWait()


#main function to handle all the requests..
def main_req():
    while True:
        req=command().lower() #call the command fnction to take user request..
        print(req)
        if "hello" in req:
            speak("Hello dear user how may i help you")
        elif "play music" in req:
            choice=random.randint(1,3)
            if choice==1:
                webbrowser.open("https://youtu.be/lBvbNxiVmZA?si=dpwY-3GIFKusbpS2")
            elif choice==2:
                webbrowser.open("https://youtu.be/pgsBaKYTi1M?si=aZhIwY26F40YUKM4")
            elif choice==3:
                webbrowser.open("https://youtu.be/_Xbpyj_v-eU?si=_eDGcSGhiBILBjEB")
            
        elif "time" in req:
            curr_time=datetime.datetime.now().strftime("%H:%M")
            speak("Current time is" +str(curr_time))
        elif "date" in req:
            curr_date=datetime.datetime.now().strftime("%d:%m")
            speak("Today's date is" +str(curr_date))
        
        #to append tasks in a file..
        elif "add task" in req:
            task=req.replace("add task","")
            task=task.strip()
            if task!="":
                speak("Adding task :"+task)
                with open("todo.txt","a") as file:
                    file.write(task)
                    file.write("\n")
        #to read out the todo list..
        elif "read list" in req:
            speak("The tasks to be performed today are ")
            with open("todo.txt","r") as file:
                speak(file.read())
                
        #to notify about the todo list..
        elif "notify" in req:
            with open("todo.txt","r") as file:
                task=file.read()
            notification.notify(title="Reminder: Today's work",message=task)
            
        #to open a specific app
        elif "open" in req:
            query=req.replace("open","")
            pyautogui.press("super") #open the windows button to search for the app..
            pyautogui.typewrite(query)
            pyautogui.sleep(4)
            pyautogui.press("enter")
            
        #to take a screenshot...
        elif "screenshot" in req:
            screenshot = pyautogui.screenshot()
            speak("Screenshot saved successfully")
            # Save it to a file
            screenshot.save("screenshot.png")
            
        #to search wikipedia..
        elif "wikipedia" in req:
            req=req.replace("search wikipedia","")
            res=wikipedia.summary(req,sentences=4)
            speak(res)
        
        
        #searching google..
        elif "search google" in req:
            req=req.replace("search google","")
            webbrowser.open("https://www.google.com/search?q="+req)
            
        #for displaying system information related to the os and hardware..
        elif "system information" in req or "system info" in req:
            uname = platform.uname()
            cpu_usage = psutil.cpu_percent(interval=1)
            ram = psutil.virtual_memory()
            battery = psutil.sensors_battery()

            info = f"""
        System: {uname.system}
        Node Name: {uname.node}
        Release: {uname.release}
        Machine: {uname.machine}
        Processor: {uname.processor}
        CPU Usage: {cpu_usage}%
        RAM Usage: {ram.percent}%
        Battery: {battery.percent if battery else 'N/A'}%
            """

            # Shortened message for notification (under 256 chars)
            short_info = f"CPU: {cpu_usage}% | RAM: {ram.percent}% | Battery: {battery.percent if battery else 'N/A'}%"

            speak("Displaying your system information now.")
            print(info)  # Full output in terminal
            notification.notify(
                title="System Information",
                message=short_info,
                timeout=10
            )

        #display the internet speed of the user's device..
        elif "internet speed" in  req:
            speak("Performing the internet speed test.")
            st = speedtest.Speedtest()
            st.get_best_server()  # Get the best server based on ping
            download_speed = round((st.download() / 1_000_000),2) # Convert from bits to megabits (Mbps)
            upload_speed = round((st.upload() / 1_000_000 ),2) # Convert from bits to megabits (Mbps)
            ping = st.results.ping  # Get ping in ms
            
            # Prepare the information to speak
            speed_info = f"""
            Download Speed: {download_speed} Mbps
            Upload Speed: {upload_speed} Mbps
            """

            # Speak and display the results
            print(speed_info)
            speak("Your download speed is {} Mbps and your upload speed is {} Mbps.".format(download_speed, upload_speed))

            notification.notify(
                title="Internet Speed Test",
                message=speed_info,
                timeout=10
            )

        #run shell commands..
        elif "run command" in req:
            speak("Voice input might not work well here. Please type the command.")
            typed_command = input("Type the shell command you want to run: ")
            try:
                result = subprocess.check_output(typed_command, shell=True, stderr=subprocess.STDOUT, text=True)
                print(result)
                speak("Command executed successfully")
                notification.notify(
                    title="Command Output",
                    message=result[:250] + "..." if len(result) > 250 else result,
                    timeout=10
                )
            except subprocess.CalledProcessError as e:
                speak("There was an error running your command.")
                print(e.output)
                    
                    
                    
        #creating powerpoint presentations....
        elif "make a presentation" in req or "powerpoint presentation" in req:
            speak("Sure. Please tell me the topic and the number of slides.")
            topic=input("Enter the topic name")
            num_slides=int(input("Enter the number of slides you want"))
            speak(f"Creating a {num_slides}-slide presentation on {topic}. Please wait.")
            slide_points = generate_slide_contents(topic, num_slides)
            create_multi_slide_presentation(topic, slide_points)
            speak(f"Presentation on '{topic}' with '{num_slides}' slides created.")



        #can also send whatsapp messages using pywhatkit.. and similarly for email..
        #for email we can use smtplib also..
        elif "stop" in req:
            speak("Wishing you a good day, take care")
            exit(0)
main_req()