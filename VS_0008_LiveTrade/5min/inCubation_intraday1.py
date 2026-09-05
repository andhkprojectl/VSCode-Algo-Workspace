from operator import truediv

import win32com.client
import time
import pytz
from datetime import datetime


class TradingState:
    def __init__(self):
        isReLoadDB5MinRun = False

state = TradingState()

# reload database to avoid bad tick
# Must run this for intraday
# 5 minute time frame, fresh 10 second, i.e.
# 04:50, 09:50, 14:50, 19:50, 24:50, 29:50, 34:50, 39:50, 44:50, 49:50, 54:50, 59:50
def reLoadDB5Min(currDateTImeUSCentral, ab, dbName1):
#    if (currDateTImeUSCentral.weekday() >= 0 and currDateTImeUSCentral.weekday() <= 4) \ # mon to fri
#        and (currDateTImeUSCentral.hour >= 9 and currDateTImeUSCentral.hour <= 16) \ # US central time, hour from 9 to 16
    if (currDateTImeUSCentral.weekday() >= 0 and currDateTImeUSCentral.weekday() <= 6) \
        and (currDateTImeUSCentral.minute % 5 == 4) \
        and (currDateTImeUSCentral.second >= 50 and currDateTImeUSCentral.second <= 59) \
        and state.isReLoadDB5MinRun == False:
        ab.LoadDatabase(dbName1)
        state.isReLoadDB5MinRun = True
        print("DataBase reloaded. ", currDateTImeUSCentral.strftime('%Y-%m-%d %H:%M:%S'))
    else:
        state.isReLoadDB5MinRun = False




# dotrade (on US Central time zone, not HKTime. To avoid change in daylight saving time)
# trade start on min 0, 30, 5, 35 (5, 35 ensure new bar occur 5 min later after 0, 30min). From 0 to 30 sec
# Mon to Fri: Trade 24 hours
# Sat, Sun: no trade
# Note for simplicity, weekly start trade may be on Sun and last trade on Fri before 00:00, above date just for testing
def isDoTrade(currDateTImeUSCentral):
    # true for Mon to Fri
    if (currDateTImeUSCentral.weekday() >= 0 and currDateTImeUSCentral.weekday() <= 4) \
        and (currDateTImeUSCentral.minute == 0 or currDateTImeUSCentral.minute == 5 or currDateTImeUSCentral.minute == 30 or currDateTImeUSCentral.minute == 35) \
        and (currDateTImeUSCentral.second >= 1 and currDateTImeUSCentral.second <= 30):
            result1 = True
    else:
            result1 = False
    # print("isDoTrade Result=", result1)            
    return result1

def isDoTrade5Min(currDateTImeUSCentral):
    # true for Mon to Fri
    if (currDateTImeUSCentral.weekday() >= 0 and currDateTImeUSCentral.weekday() <= 4) \
        and (0 <= currDateTImeUSCentral.minute <= 55 and currDateTImeUSCentral.minute % 5 == 0) \
        and (currDateTImeUSCentral.second >= 1 and currDateTImeUSCentral.second <= 40):
            result1 = True
    else:
            result1 = False
    # print("isDoTrade Result=", result1)
    return result1

# and (currDateTImeUSCentral.minute == 28 or currDateTImeUSCentral.minute == 58) \
def isDoTradeUsingCPrice(currDateTImeUSCentral):
    # true for Mon to Fri
    if (currDateTImeUSCentral.weekday() >= 0 and currDateTImeUSCentral.weekday() <= 4) \
            and (currDateTImeUSCentral.minute == 29 or currDateTImeUSCentral.minute == 59) \
        and (currDateTImeUSCentral.second >= 1 and currDateTImeUSCentral.second <= 30):
            result1 = True
    else:
            result1 = False
    # print("isDoTrade Result=", result1)
    return result1
    

def run_amibroker_analysis():
    isDebug = False
    # Create one instance

    try:
        # Create an instance of the AmiBroker application
        amibroker = win32com.client.Dispatch("Broker.Application")

        # below DB Name for reload DB. Remove bad tick
        # dbName = "C:\\Program Files\\AmiBroker\\Databases\\DTNDb5_1_min"
        dbName = "C:\\Program Files\\AmiBroker\\Databases\\DTNDb5_5_min"
        
        # Open the Analysis Window
        # analysis = amibroker.Analysis

        # Load the Analysis Project
        # apx_file_path_YM = "C:\\Project\\ProjectLife\\amibroker\\system\\PL_TD_Ema_20241023_1.0.0_YM\\5 Incubation\\30min\\inCubation_YM_30min.apx"
        # analysisDocYM = amibroker.analysisDocs.open(apx_file_path_YM)

        # apx_file_path_YM = "C:\\Project\\ProjectLife\\amibroker\\system\\PL_20250914_1.0.0_YM\\5 Incubation\\inCubation_YM.apx"
        # analysisDocYM = amibroker.analysisDocs.open(apx_file_path_YM)

        # apx_file_path_YM2 = "C:\\Project\\ProjectLife\\amibroker\\system\\PL_20251019_1.0.0_YM\\5 Incubation\\inCubation_YM2.apx"
        # analysisDocYM2 = amibroker.analysisDocs.open(apx_file_path_YM2)

        # apx_file_path_NQ = "C:\\Project\\ProjectLife\\amibroker\\system\\PL_20250922_1.0.0_NQ\\5 Incubation\\inCubation_NQ.apx"
        # analysisDocNQ = amibroker.analysisDocs.open(apx_file_path_NQ)

        # apx_file_path_NQ2 = "C:\\Project\\ProjectLife\\amibroker\\system\\PL_20251018_1.0.0_NQ\\5 Incubation\\inCubation_NQ2.apx"
        # analysisDocNQ2 = amibroker.analysisDocs.open(apx_file_path_NQ2)

        apx_file_path_nvda1 = "C:\\Project\\ProjectLife\\amibroker\\system\\PL_20260204_1.0.0_NVDA_100\\5 Incubation\\inCubation_NVDA_100.apx"
        analysisDocNVDA1 = amibroker.analysisDocs.open(apx_file_path_nvda1)

        # apx_file_path_QMGC = "C:\\Project\\ProjectLife\\amibroker\\system\\PL_TD_Alligator_20240728_1.0.0_QMGC\\6 Incubation\\inCubation_QMGC.apx"
        # analysisDocQMGC = amibroker.analysisDocs.open(apx_file_path_QMGC)
        # apx_file_path_ZS = "C:\\Project\\ProjectLife\\amibroker\\system\\PL_TD_Ema_20241023_1.0.0_S\\5 Incubation\\30min\\inCubation_ZS_30min.apx"
        # analysisDocZS = amibroker.analysisDocs.open( apx_file_path_ZS)

        # apx_file_path_DIA = "C:\\Project\\ProjectLife\\amibroker\\system\\PL_20250105_1.0.0_DIA\\5 Incubation\\30min\\inCubation_DIA_30min.apx"
        # analysisDocDIA = amibroker.analysisDocs.open( apx_file_path_DIA)

        # apx_file_path_QC = "C:\\Project\\ProjectLife\\amibroker\\system\\PL_20241227_1.0.0_QC\\5 Incubation\\30min\\inCubation_QC_30min.apx"
        # analysisDocQC = amibroker.analysisDocs.open( apx_file_path_QC)

        # apx_file_path_QM = "C:\\Project\\ProjectLife\\amibroker\\system\\PL_20241213_1.0.0_QM\\5 Incubation\\30min\\inCubation_QM_30min.apx"
        # analysisDocQM = amibroker.analysisDocs.open( apx_file_path_QM)


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

            reLoadDB5Min(central_time_now, amibroker, dbName)

            # Check if it's Monday and the time is 2:00 AM
            #if now.weekday() == 0 and now.hour == 2 and now.minute == 0:
            #    print("Hello, World!")
            #    time.sleep(60)  # Wait for a minute to avoid multiple prints
            # isDoTrade mean buy / short using ref(buy or short, -1) and buyPrice or shortPrice is O
            if (isDoTrade(central_time_now) == True):
                print("isDoTrade is true, do trade")
                print("Current Hong Kong Time:", hong_kong_time_now.strftime('%Y-%m-%d %H:%M:%S'))
                print("Converted US Central Time:", central_time_now.strftime('%Y-%m-%d %H:%M:%S'))
                
                # Run the Analysis
                # analysis.Run(1)  # 1 means to run in the foreground
                # analysis.Backtest() # backtest

                #
                # QMGC begin
                #
                # analysisDocQMGC.run(2)  # backtest
                # print("QMGC. After run(2):", hong_kong_time_now.strftime('%Y-%m-%d %H:%M:%S'))

                # while analysisDocQMGC.IsBusy:
                #    time.sleep(1)

                # print("QMGC. After isBusy for loop: ", hong_kong_time_now.strftime('%Y-%m-%d %H:%M:%S'))
                #
                # QMGC end
                #

                #
                # ZS begin
                #
                # analysisDocZS.run(2)  # backtest

                # print("ZS. After run(2):", hong_kong_time_now.strftime('%Y-%m-%d %H:%M:%S'))

                # while analysisDocZS.IsBusy:
                #    time.sleep(1)

                #
                # ZS end
                #



                #
                # DIA begin
                #
                # analysisDocDIA.run(2)  # backtest

                # print("DIA. After run(2):", hong_kong_time_now.strftime('%Y-%m-%d %H:%M:%S'))

                # while analysisDocDIA.IsBusy:
                #    time.sleep(1)

                #
                # DIA end
                #

                # print("ZS. After isBusy for loop: ", hong_kong_time_now.strftime('%Y-%m-%d %H:%M:%S'))

            # if (isDoTrade5Min(central_time_now) == True and state.isReLoadDB5MinRun == False):
            #     amibroker.LoadDatabase(dbName)
            #     state.isReLoadDB5MinRun = True
            #     print("DataBase reloaded. ", central_time_now.strftime('%Y-%m-%d %H:%M:%S'))
            # else:
            #     state.isReLoadDB5MinRun = False

            if (isDoTrade5Min(central_time_now) == True):
                #
                # YM begin
                #
                # analysisDocYM.run(2)  # backtest
                #
                # print("YM. After run(2):", hong_kong_time_now.strftime('%Y-%m-%d %H:%M:%S'))
                #
                # while analysisDocYM.IsBusy:
                #     time.sleep(1)
                #
                # print("YM. After isBusy for loop: ", hong_kong_time_now.strftime('%Y-%m-%d %H:%M:%S'))
                #
                # YM end
                #

                #
                # YM2 begin
                #
                # analysisDocYM2.run(2) # backtest
                #
                # print("YM2. After run(2):", hong_kong_time_now.strftime('%Y-%m-%d %H:%M:%S'))
                #
                # while analysisDocYM2.IsBusy:
                #   time.sleep(1)
                #
                # print("YM2. After isBusy for loop: ", hong_kong_time_now.strftime('%Y-%m-%d %H:%M:%S'))

                #
                # YM2 end
                #

                #
                # NQ begin
                #
                # analysisDocNQ.run(2) # backtest
                #
                # print("NQ. After run(2):", hong_kong_time_now.strftime('%Y-%m-%d %H:%M:%S'))
                #
                # while analysisDocNQ.IsBusy:
                #   time.sleep(1)
                #
                # print("NQ. After isBusy for loop: ", hong_kong_time_now.strftime('%Y-%m-%d %H:%M:%S'))

                #
                # NQ end
                #


                #
                # NQ2 begin
                #
                # analysisDocNQ2.run(2) # backtest
                #
                # print("NQ2. After run(2):", hong_kong_time_now.strftime('%Y-%m-%d %H:%M:%S'))
                #
                # while analysisDocNQ2.IsBusy:
                #   time.sleep(1)
                #
                # print("NQ2. After isBusy for loop: ", hong_kong_time_now.strftime('%Y-%m-%d %H:%M:%S'))

                #
                # NQ2 end
                #


                # NVDA1 begin

                analysisDocNVDA1.run(2) # backtest

                print("NVDA1. After run(2):", hong_kong_time_now.strftime('%Y-%m-%d %H:%M:%S'))

                while analysisDocNVDA1.IsBusy:
                  time.sleep(1)

                print("NQ. After isBusy for loop: ", hong_kong_time_now.strftime('%Y-%m-%d %H:%M:%S'))

                #
                # NVDA1 end
                #


            # isDoTradeUsingCPrice mean buy / short using buy or short and buyPrice or shortPrice is O
            if (isDoTradeUsingCPrice(central_time_now) == True):
                print("isDoTrade is true, do trade")
                print("Current Hong Kong Time:", hong_kong_time_now.strftime('%Y-%m-%d %H:%M:%S'))
                print("Converted US Central Time:", central_time_now.strftime('%Y-%m-%d %H:%M:%S'))

                #
                # QC begin
                #
                # analysisDocQC.run(2)  # backtest
                #
                # print("QC. After run(2):", hong_kong_time_now.strftime('%Y-%m-%d %H:%M:%S'))
                #
                # while analysisDocQC.IsBusy:
                #    time.sleep(1)

                #
                # QC end
                #

                #
                # QM begin
                #
                # analysisDocQM.run(2)  # backtest
                #
                # print("QM. After run(2):", hong_kong_time_now.strftime('%Y-%m-%d %H:%M:%S'))
                #
                # while analysisDocQM.IsBusy:
                #     time.sleep(1)

                #
                # QM end
                #

            # Sleep for a short time to prevent high CPU usage
            time.sleep(5)


        # amibroker.quit()  # close whole amibroker

        print("Analysis project executed successfully.")
    except KeyboardInterrupt:
        # This is optional; the signal handler will catch the interrupt
        print('Keyboard interrupt caught, close analysisDoc')

        # analysisDocQMGC.close()
        # analysisDocZS.close()
        # analysisDocQM.close()
        # analysisDocQC.close()
        # analysisDocYM.close()
        # analysisDocYM2.close()
        # analysisDocNQ.close()
        # analysisDocNQ2.close()
        # analysisDocDIA.close()
        analysisDocNVDA1.close()

    except Exception as e:
        print(f"An error occurred: {e}")

# Replace with the path to your .apx file
run_amibroker_analysis()


