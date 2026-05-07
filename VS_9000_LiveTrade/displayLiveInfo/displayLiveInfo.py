import win32com.client
import time
import pytz
from datetime import datetime

# true 20 minutes 3 times every hour from Monday to Fri
def isDisplayLiveInfo(currDateTImeUSCentral):
    # true for Mon to Fri
    # >= 1 and <= 60 mean all seconds
    if (currDateTImeUSCentral.weekday() >= 0 and currDateTImeUSCentral.weekday() <= 4) \
        and (currDateTImeUSCentral.minute == 8 or currDateTImeUSCentral.minute == 28 or currDateTImeUSCentral.minute == 48) \
        and (currDateTImeUSCentral.second >= 1 and currDateTImeUSCentral.second <= 60):
            result1 = True
    else:
            result1 = False
    # print("isDisplayLiveInfo Result=", result1)
    return result1
    

def run_amibroker_analysis():
    isDebug = False
    try:
        # Create an instance of the AmiBroker application
        amibroker = win32com.client.Dispatch("Broker.Application")
        
        # Open the Analysis Window
        # analysis = amibroker.Analysis

        # Load the Analysis Project
        apx_file_path_dspLiveInfo = "C:\\Project\\ProjectLife\\amibroker\\autotrade\\prod\\apx\\displayIBLiveInfo.apx"
        analysisDocDspLiveInfo = amibroker.analysisDocs.open(apx_file_path_dspLiveInfo)

        i = 0
        while True:
        # while (i < 5):
            # i = i+1
            
            # Get the current day and time
            # now = datetime.now()

            ## convert to us central time
            # Define time zones
            hong_kong_tz = pytz.timezone('Asia/Hong_Kong')
            central_tz = pytz.timezone('US/Eastern')

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
            if (isDisplayLiveInfo(central_time_now) == True):
                print("isDisplayLiveInfo is true, do trade")
                print("Current Hong Kong Time:", hong_kong_time_now.strftime('%Y-%m-%d %H:%M:%S'))
                print("Converted US Central Time:", central_time_now.strftime('%Y-%m-%d %H:%M:%S'))
                
                # Run the Analysis
                # analysis.Run(1)  # 1 means to run in the foreground
                # analysis.Backtest() # backtest

                #
                # display Live Info begin
                #
                analysisDocDspLiveInfo.run(1) # explore

                print("displayLiveInfo. After run(1):", hong_kong_time_now.strftime('%Y-%m-%d %H:%M:%S'))
                        
                while analysisDocDspLiveInfo.IsBusy:
                  time.sleep(1)

                print("displayLiveInfo. After isBusy for loop: ", hong_kong_time_now.strftime('%Y-%m-%d %H:%M:%S'))
                #
                # display Live Info  end
                #

            # Sleep for a short time to prevent high CPU usage
            time.sleep(40)


        # amibroker.quit()  # close whole amibroker

        print("Analysis project executed successfully.")
    except KeyboardInterrupt:
        # This is optional; the signal handler will catch the interrupt
        print('Keyboard interrupt caught, close analysisDoc')
        analysisDocDspLiveInfo.close()

    except Exception as e:
        print(f"An error occurred: {e}")

# Replace with the path to your .apx file
run_amibroker_analysis()


ib = IB()
#  directly connect to TWS from python
# Connect to TWS or IB Gateway
# Default TWS port is 7497, default clientId is 1
try:
    ib.connect(host='127.0.0.1', port=7497, clientId=1) # tws client
    # ib.connect(host='127.0.0.1', port=4002, clientId=1)  # IB gateway
    print("Connected to Interactive Brokers.")
except Exception as e:
    print(f"Failed to connect: {e}")
    # exit()

# Fetch open positions
try:
    positions = ib.positions()
    print("Open positions:")
    if positions:
        for pos in positions:
            symbol = pos.contract.symbol
            position_size = pos.position
            print(f"Symbol: {symbol}, Shares: {position_size}")
    else:
        print("No open positions.")
except Exception as e:
    print(f"Error fetching positions: {e}")

# Disconnect from TWS
ib.disconnect()


