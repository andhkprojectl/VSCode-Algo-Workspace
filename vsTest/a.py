import win32com.client
import time
import pytz
from datetime import datetime


# dotrade (on US Central time zone, not HKTime. To avoid change in daylight saving time)
# Mon to Thu: 00:00:01 to 00:00:30 or 00:05:01 to 00:05:30, every 5 second (new bar after 00:00:00)
# Sun: 20:00:01 to 20:00:30 or 20:05:01 to 20:05:30. Start trading on Sun 20:00 (new bar after 20:00:00)
# Fri: 14:15:01 to 14:15:30 or 14:28:01 to 14:28:30 (14:20:00 end trade)
# Sat: no trade
def isDoTrade(currDateTImeUSCentral):
    # True Sun 
    if currDateTImeUSCentral.weekday() == 6 \
           and (currDateTImeUSCentral.hour == 20 and (currDateTImeUSCentral.minute == 0 or currDateTImeUSCentral.minute == 5) and currDateTImeUSCentral.second >= 1 and currDateTImeUSCentral.second <= 30):
            result1 = True
    # True if Fri
    elif currDateTImeUSCentral.weekday() == 4 \
            and (currDateTImeUSCentral.hour == 14 and (currDateTImeUSCentral.minute == 15 or currDateTImeUSCentral.minute == 18) and currDateTImeUSCentral.second >= 1 and currDateTImeUSCentral.second <= 30):
            result1 = True
    # True if Mon, Tue, Wed, Thu
    elif (currDateTImeUSCentral.weekday() == 0 or currDateTImeUSCentral.weekday() == 1 or currDateTImeUSCentral.weekday() == 2 or currDateTImeUSCentral.weekday() == 3) \
            and (currDateTImeUSCentral.hour == 8 and (currDateTImeUSCentral.minute == 9 or currDateTImeUSCentral.minute == 10) and currDateTImeUSCentral.second >= 1 and currDateTImeUSCentral.second <= 30):
            result1 = True
    # test
    #elif (currDateTImeUSCentral.weekday() == 0 or currDateTImeUSCentral.weekday() == 1 or currDateTImeUSCentral.weekday() == 2 or currDateTImeUSCentral.weekday() == 3) \
    #        and (currDateTImeUSCentral.hour == 8 and (currDateTImeUSCentral.minute == 20 or currDateTImeUSCentral.minute == 25) and currDateTImeUSCentral.second >= 1 and currDateTImeUSCentral.second <= 30):
    #        result1 = True            
    else:
            result1 = False
    # print("isDoTrade Result=", result1)            
    return result1
    

def run_amibroker_analysis(apx_file_path):
    isDebug = True
    try:
        # Create an instance of the AmiBroker application
        amibroker = win32com.client.Dispatch("Broker.Application")
        
        # Open the Analysis Window
        # analysis = amibroker.Analysis

        # Load the Analysis Project
        # analysis.Load(apx_file_path)
        analysisDocs1 = amibroker.analysisDocs.open( apx_file_path)

        i = 0
        # while True:
        while (i < 5):
            i = i+1
            
            # Get the current day and time
            # now = datetime.now()

            ## convert to us central time
            # Define time zones
            hong_kong_tz = pytz.timezone('Asia/Hong_Kong')
            central_tz = pytz.timezone('US/Central')

            # Get current time in Hong Kong
            hong_kong_time_now = datetime.now(hong_kong_tz)

            # Convert to US Central Time
            central_time_now = hong_kong_time_now.astimezone(central_tz)

            # Display the results
            if (isDebug==True):
                print("Current Hong Kong Time:", hong_kong_time_now.strftime('%Y-%m-%d %H:%M:%S'))
                print("Converted US Central Time:", central_time_now.strftime('%Y-%m-%d %H:%M:%S'))
                print("weekday:" + str(central_time_now.weekday()))
                print("hour:" + str(central_time_now.hour))
                print("minute:" + str(central_time_now.minute))
                print("second:" + str(central_time_now.second))
                
            if (central_time_now.minute == 0 or central_time_now.minute == 5):
                print("5.Current Hong Kong Time:", hong_kong_time_now.strftime('%Y-%m-%d %H:%M:%S'))
                print("5. Converted US Central Time:", central_time_now.strftime('%Y-%m-%d %H:%M:%S'))                

            # Check if it's Monday and the time is 2:00 AM
            #if now.weekday() == 0 and now.hour == 2 and now.minute == 0:
            #    print("Hello, World!")
            #    time.sleep(60)  # Wait for a minute to avoid multiple prints
            if (isDoTrade(central_time_now) == True):
                print("isDoTrade is true, do trade")
                print("Current Hong Kong Time:", hong_kong_time_now.strftime('%Y-%m-%d %H:%M:%S'))
                print("Converted US Central Time:", central_time_now.strftime('%Y-%m-%d %H:%M:%S'))
                
                # Run the Analysis
                # analysis.Run(1)  # 1 means to run in the foreground
                # analysis.Backtest() # backtest
                analysisDocs1.run(2) # backtest
				
                print("After run(2):", hong_kong_time_now.strftime('%Y-%m-%d %H:%M:%S'))
                        
                while analysisDocs1.IsBusy:
                  time.sleep(1)
				 
                print("After isBusy for loop: ", hong_kong_time_now.strftime('%Y-%m-%d %H:%M:%S'))

            # Sleep for a short time to prevent high CPU usage
            time.sleep(5)

            
        
		
        analysisDocs1.close()
		
        # amibroker.quit()  # close whole amibroker

        print("Analysis project executed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Replace with the path to your .apx file
apx_file_path = "C:\\Project\\ProjectLife\\amibroker\\system\\PL_TD_Ema_20241023_1.0.0_S\\5 Incubation\\inCubation_ZS.apx"
run_amibroker_analysis(apx_file_path)
