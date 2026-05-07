import win32com.client
import time


def reset_iqfeed_connection():
    try:
        # Connect to the running AmiBroker instance
        ab = win32com.client.Dispatch("Broker.Application")

        # dbName = "C:\\Program Files\\AmiBroker\\Databases\\DTNDb5_1_min"
        dbName = "C:\\Program Files\\AmiBroker\\Databases\\DTNDb5_5_min"

        ab.LoadDatabase(dbName)
        # ab.refreshall()

        # print("Sending shutdown command to IQFeed...")
        # Most plugins respond to 'shutdown' or 'disconnect'
        # ab.ExecuteBatch("DataPluginCmd iqfeed shutdown")

        # time.sleep(2)  # Brief pause

        # print("Sending reconnect command...")
        # 'reconnect' is the standard command for most RT plugins
        # ab.ExecuteBatch("DataPluginCmd iqfeed reconnect")

        print("Connection reset signal sent.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    reset_iqfeed_connection()