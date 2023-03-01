import logging
import os
import tarfile
import time
from datetime import datetime

import boto3
import botocore


def setup_logging(default_level=logging.WARNING):
    """
    Setup logging configuration
    """
    logging.basicConfig(level=default_level)
    return logging.getLogger('DeployPySparkScriptOnAws')


def terminate(error_message=None):
    """
    Method to exit the Python script. It will log the given message and then exit().
    :param error_message:
    """
    if error_message:
        logger.error(error_message)
    logger.critical('The script is now terminating')
    exit()


class DeployPySparkScriptOnAws:
    """
    Programmatically deploy a local PySpark script on an AWS cluster
    """

    def __init__(self):
        self.app_name = "word_count_spark"                  # Application name
        self.ec2_key_name = "e2_key_name"                   # Key name to use for cluster
        self.job_flow_id = None                             # Returned by AWS in start_spark_cluster()
        self.job_name = None                                # Filled by generate_job_name()
        self.path_script = "spark_example/"                 # Path of Spark script to be deployed on AWS Cluster
        self.s3_bucket_logs = "aws-logs-XXXXXX-eu-west-1"   # S3 Bucket to store AWS EMR logs
        self.s3_bucket_temp_files = "bucket-temp-files"     # S3 Bucket to store temporary files
        self.s3_region = 's3-eu-west-1.amazonaws.com'       # S3 region to specifiy s3Endpoint in s3-dist-cp step
        self.user = 'thom'                                  # Define user name

    def run(self):
        session = boto3.Session(profile_name='thom')        # Select AWS IAM profile
        s3 = session.resource('s3')                         # Open S3 connection
        self.generate_job_name()                            # Generate job name
        self.temp_bucket_exists(s3)                         # Check if S3 bucket to store temporary files in exists
        self.tar_python_script()                            # Tar the Python Spark script
        self.upload_temp_files(s3)                          # Move the Spark files to a S3 bucket for temporary files
        c = session.client('emr')                           # Open EMR connection
        self.start_spark_cluster(c)                         # Start Spark EMR cluster
        self.step_spark_submit(c)                           # Add step 'spark-submit'
        self.describe_status_until_terminated(c)            # Describe cluster status until terminated
        self.remove_temp_files(s3)                          # Remove files from the temporary files S3 bucket

    def generate_job_name(self):
        self.job_name = "{}.{}.{}".format(self.app_name,
                                          self.user,
                                          datetime.now().strftime("%Y%m%d.%H%M%S.%f"))

    def temp_bucket_exists(self, s3):
        """
        Check if the bucket we are going to use for temporary files exists.
        :param s3:
        :return:
        """
        try:
            s3.meta.client.head_bucket(Bucket=self.s3_bucket_temp_files)
        except botocore.exceptions.ClientError as e:
            # If a client error is thrown, then check that it was a 404 error.
            # If it was a 404 error, then the bucket does not exist.
            error_code = int(e.response['Error']['Code'])
            if error_code == 404:
                terminate("Bucket for temporary files does not exist")
            terminate("Error while connecting to Bucket")
        logger.info("S3 bucket for temporary files exists")

    def tar_python_script(self):
        """

        :return:
        """
        # Create tar.gz file
        t_file = tarfile.open("files/script.tar.gz", 'w:gz')
        # Add Spark script path to tar.gz file
        files = os.listdir(self.path_script)
        for f in files:
            t_file.add(self.path_script + f, arcname=f)
        # List all files in tar.gz
        for f in t_file.getnames():
            logger.info("Added %s to tar-file" % f)
        t_file.close()

    def upload_temp_files(self, s3):
        """
        Move the PySpark script files to the S3 bucket we use to store temporary files
        :param s3:
        :return:
        """
        # Shell file: setup (download S3 files to local machine)
        s3.Object(self.s3_bucket_temp_files, self.job_name + '/setup.sh')\
          .put(Body=open('files/setup.sh', 'rb'), ContentType='text/x-sh')
        # Shell file: Terminate idle cluster
        s3.Object(self.s3_bucket_temp_files, self.job_name + '/terminate_idle_cluster.sh')\
          .put(Body=open('files/terminate_idle_cluster.sh', 'rb'), ContentType='text/x-sh')
        # Compressed Python script files (tar.gz)
        s3.Object(self.s3_bucket_temp_files, self.job_name + '/script.tar.gz')\
          .put(Body=open('files/script.tar.gz', 'rb'), ContentType='application/x-tar')
        logger.info(f"Uploaded files to key '{self.job_name}' in bucket '{self.s3_bucket_temp_files}'")
        return True

    def remove_temp_files(self, s3):
        """
        Remove Spark files from temporary bucket
        :param s3:
        :return:
        """
        bucket = s3.Bucket(self.s3_bucket_temp_files)
        for key in bucket.objects.all():
            if key.key.startswith(self.job_name) is True:
                key.delete()
                logger.info(f"Removed '{key.key}' from bucket for temporary files")

    def start_spark_cluster(self, c):
        """

        :param c: EMR client
        :return:
        """
        response = c.run_job_flow(
            Name=self.job_name,
            LogUri=f"s3://{self.s3_bucket_logs}/elasticmapreduce/",
            ReleaseLabel="emr-4.4.0",
            Instances={
                'InstanceGroups': [
                    {
                        'Name': 'EmrMaster',
                        'Market': 'SPOT',
                        'InstanceRole': 'MASTER',
                        'BidPrice': '0.05',
                        'InstanceType': 'm3.xlarge',
                        'InstanceCount': 1,
                    },
                    {
                        'Name': 'EmrCore',
                        'Market': 'SPOT',
                        'InstanceRole': 'CORE',
                        'BidPrice': '0.05',
                        'InstanceType': 'm3.xlarge',
                        'InstanceCount': 2,
                    },
                ],
                'Ec2KeyName': self.ec2_key_name,
                'KeepJobFlowAliveWhenNoSteps': False
            },
            Applications=[{'Name': 'Hadoop'}, {'Name': 'Spark'}],
            JobFlowRole='EMR_EC2_DefaultRole',
            ServiceRole='EMR_DefaultRole',
            VisibleToAllUsers=True,
            BootstrapActions=[
                {
                    'Name': 'setup',
                    'ScriptBootstrapAction': {
                        'Path': f's3n://{self.s3_bucket_temp_files}/{self.job_name}/setup.sh'.format(),
                        'Args': [
                            f's3://{self.s3_bucket_temp_files}/{ self.job_name}',
                        ]
                    }
                },
                {
                    'Name': 'idle timeout',
                    'ScriptBootstrapAction': {
                        'Path': f's3n://{self.s3_bucket_temp_files}/{self.job_name}/terminate_idle_cluster.sh',
                        'Args': ['3600', '300']
                    }
                },
            ],
        )
        # Process response to determine if Spark cluster was started, and if so, the JobFlowId of the cluster
        response_code = response['ResponseMetadata']['HTTPStatusCode']
        if response['ResponseMetadata']['HTTPStatusCode'] == 200:
            self.job_flow_id = response['JobFlowId']
        else:
            terminate(f"Could not create EMR cluster (status code {response_code})")

        logger.info(f"Created Spark EMR-4.4.0 cluster with JobFlowId {self.job_flow_id}")

    def describe_status_until_terminated(self, c):
        """
        :param c:
        :return:
        """
        stop = False
        while stop is False:
            description = c.describe_cluster(ClusterId=self.job_flow_id)
            state = description['Cluster']['Status']['State']
            if state == 'TERMINATED' or state == 'TERMINATED_WITH_ERRORS':
                stop = True
            logger.info(state)
            time.sleep(30)  # Prevent ThrottlingException by limiting number of requests

    def step_spark_submit(self, c, arguments):
        """

        :param c:
        :return:
        """
        c.add_job_flow_steps(
            JobFlowId=self.job_flow_id,
            Steps=[
                {
                    'Name': 'Spark Application',
                    'ActionOnFailure': 'CONTINUE',
                    'HadoopJarStep': {
                        'Jar': 'command-runner.jar',
                        'Args': [
                            "spark-submit",
                            "/home/hadoop/wordcount.py",
                            arguments
                        ]
                    }
                },
            ]
        )
        logger.info(f"Added step 'spark-submit' with argument '{arguments}'")
        time.sleep(1)  # Prevent ThrottlingException

    def step_copy_data_between_s3_and_hdfs(self, c, src, dest):
        """
        Copy data between S3 and HDFS (not used for now)
        :param c:
        :return:
        """
        c.add_job_flow_steps(
            JobFlowId=self.job_flow_id,
            Steps=[{
                    'Name': 'Copy data from S3 to HDFS',
                    'ActionOnFailure': 'CANCEL_AND_WAIT',
                    'HadoopJarStep': {
                        'Jar': 'command-runner.jar',
                        'Args': [
                            "s3-dist-cp",
                            "--s3Endpoint=s3-eu-west-1.amazonaws.com",
                            f"--src={src}",
                            f"--dest={dest}"
                        ]
                    }
                }]
        )
        logger.info(f"Added step 'Copy data from {src} to {dest}'")


logger = setup_logging()

if __name__ == "__main__":
    DeployPySparkScriptOnAws().run()
