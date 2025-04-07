---
title: "Hosting Onion v3 Websites on SaaS Platforms"
date: 2025-04-07
id: 8
author: "Afnan K Salal"
authorGithub: "https://github.com/afnanksalal"
tags:
  - Tor
  - Onion v3
  - Hidden Services
  - SaaS
  - Koyeb
  - Render
  - Security
---

## **Hosting Onion v3 Websites on SaaS Platforms**

### **1. Introduction to Tor and Onion Services**

The Tor network provides anonymity and privacy by routing internet traffic through a distributed network of relays.  This makes it difficult to trace the origin or destination of the traffic. A key feature of Tor is its support for Onion Services (formerly known as Hidden Services). Onion Services allow users to host websites and other services that are only accessible through the Tor network, adding an extra layer of privacy and security.

#### **1.1 The Power of Anonymity: Tor's Role**

Tor shields users from network surveillance by encrypting and routing their traffic through a series of volunteer-operated relays. This masks the user's IP address and makes it difficult for eavesdroppers to monitor their online activities.  Tor is essential for activists, journalists, whistleblowers, and anyone who values their online privacy.

#### **1.2 Onion Services: Anonymous Hosting**

Onion Services allow individuals and organizations to host websites, chat servers, and other services anonymously. These services are identified by a `.onion` address, which is cryptographically derived from the service's public key. Unlike regular websites, Onion Services don't rely on publicly visible IP addresses or domain names, making them much harder to censor or track.

### **2. Onion v3: Evolution and Security**

Onion v3 is the latest iteration of the Onion Services protocol. It offers significant improvements over the older Onion v2 protocol in terms of security, usability, and address length.

#### **2.1 Why Upgrade to Onion v3?**

Onion v2 addresses were susceptible to various vulnerabilities, including collision attacks and length limitations. Onion v3 addresses are:

*   **Longer**: 56-character addresses providing a vastly larger address space (2<sup>224</sup> possible addresses), making collision attacks practically infeasible.
*   **More Secure**: Uses state-of-the-art cryptography (Curve25519, SHA3, and Ed25519) to enhance security.
*   **More User-Friendly**: The larger address space allows for more descriptive and memorable addresses.

The Tor Project officially deprecated Onion v2 in October 2021, strongly urging all service operators to migrate to Onion v3.

#### **2.2 Understanding the Onion v3 Address Structure**

An Onion v3 address is a 56-character string, such as `exampleabc123defghi456jklmno789pqrstu012vwxyz.onion`. This address is derived cryptographically from the service's public key and includes a checksum to ensure its integrity. When a user enters an Onion v3 address into the Tor Browser, it initiates a rendezvous process that establishes a secure connection to the hidden service.

### **3. Hosting Onion Websites on SaaS Platforms**

Traditionally, hosting an Onion Service required setting up a dedicated server and configuring Tor manually.  However, with the rise of SaaS (Software as a Service) platforms, it's now possible to deploy Onion Services more easily and efficiently. Platforms like Koyeb and Render offer features that simplify the process, although some manual configuration might still be needed.

#### **3.1 Challenges and Considerations**

Hosting an Onion Service on a SaaS platform presents unique challenges:

*   **Tor Installation and Configuration**: Most SaaS platforms don't have Tor pre-installed, requiring you to install and configure it within your deployment.
*   **Port Exposure**: Onion Services typically use internal ports (e.g., `127.0.0.1:8080`), which need to be properly exposed and managed within the SaaS environment.
*   **File System Access**: Accessing and managing the Onion Service's key files (usually located in `/var/lib/tor/onion_service/`) might be restricted on some platforms.
*   **Logging and Monitoring**:  You'll need to configure Tor's logging and monitoring to ensure the service is running correctly.

#### **3.2 Using Docker for Reproducibility and Portability**

Docker is an excellent tool for packaging your Onion Service and its dependencies into a portable container. The provided Dockerfile is a solid starting point for deploying an Onion Service on various SaaS platforms. Here's a breakdown of the Dockerfile and explanations:

```dockerfile
# Use a minimal base image with Tor and Nginx pre-installed
FROM ubuntu:22.04

# Set environment variables to avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary packages: Tor, Nginx, build tools, libsodium-dev, and Python for the dummy server
RUN apt-get update && \
    apt-get install -y tor nginx curl build-essential libssl-dev git autoconf automake libtool libsodium-dev python3 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Clone and build mkp224o for generating vanity .onion addresses
RUN git clone https://github.com/cathugger/mkp224o.git /mkp224o && \
    cd /mkp224o && \
    ./autogen.sh && \
    ./configure && \
    make

# Copy the static website files into the container
COPY ./static /var/www/html/

# Configure Nginx to serve the static website
RUN echo "server { \
            listen 127.0.0.1:8080; \
            root /var/www/html; \
            index index.html; \
            location / { try_files \$uri \$uri/ =404; } \
        }" > /etc/nginx/sites-available/default && \
    ln -sf /etc/nginx/sites-available/default /etc/nginx/sites-enabled/default

# Generate a single custom vanity .onion address using mkp224o
RUN mkdir -p /var/lib/tor/onion_service && \
    /mkp224o/mkp224o -d /var/lib/tor/onion_service -T 3 -n 1 yourpatternhere && \
    chmod -R 700 /var/lib/tor/onion_service

# Fix ownership of the onion_service directory for the debian-tor user
RUN chown -R debian-tor:debian-tor /var/lib/tor/onion_service

# Configure Tor to expose the Onion Service and log to a file
RUN echo "HiddenServiceDir /var/lib/tor/onion_service/ \n\
          HiddenServicePort 80 127.0.0.1:8080 \n\
          Log notice file /var/log/tor/log" >> /etc/tor/torrc

# Create the Tor log directory and file, and fix ownership/permissions
RUN mkdir -p /var/log/tor && \
    touch /var/log/tor/log && \
    chown -R debian-tor:debian-tor /var/log/tor && \
    chmod -R 640 /var/log/tor/log

# Create a dummy HTML file for the fallback server
RUN mkdir -p /var/www/dummy && \
    echo "<html><body><h1>This is a dummy server. The real service might not be running.</h1></body></html>" > /var/www/dummy/index.html

# Expose the Onion Service hostname to the host machine
VOLUME /var/lib/tor/onion_service

# Start Nginx, Tor, and the dummy server when the container starts
CMD service nginx start && \
    service tor start && \
    echo "Your .onion address is:" && \
    cat /var/lib/tor/onion_service/hostname && \
    (cd /var/www/dummy && python3 -m http.server 8081) & \
    tail -f /var/log/tor/log
```

*   **`FROM ubuntu:22.04`**:  Specifies the base image as Ubuntu 22.04.  This provides a stable and familiar environment.
*   **`RUN apt-get update && apt-get install -y ...`**: Installs the necessary packages, including `tor`, `nginx`, and `libsodium-dev`.  `nginx` serves the website content, and `libsodium-dev` is needed for `mkp224o`.
*   **`RUN git clone ... && cd /mkp224o && ./autogen.sh ...`**: Clones, builds, and installs `mkp224o`, a tool for generating vanity Onion addresses (addresses that contain specific keywords).  **Important:**  Generating a vanity address significantly increases the deployment time. Consider removing this step if you don't need a custom address. Also, generating a vanity address is resource intensive, especially for more complex prefixes.
*   **`COPY ./static /var/www/html/`**: Copies your website's static files into the Nginx document root.
*   **`RUN echo "server { ... }" > /etc/nginx/sites-available/default ...`**: Configures Nginx to serve the static website on `127.0.0.1:8080`.  This internal port is then exposed via the Tor Onion Service.
*   **`RUN mkdir -p /var/lib/tor/onion_service && /mkp224o/mkp224o ...`**:  Generates the Onion Service keypair and hostname using `mkp224o`. This is where the custom address prefix (`yourpatternhere`) is specified.  If you skip the vanity address generation, you'll need to use `tor --keygen HiddenServiceDir /var/lib/tor/onion_service/` instead.
*   **`RUN chown -R debian-tor:debian-tor /var/lib/tor/onion_service`**: Sets the correct ownership for the Onion Service directory, ensuring that the `debian-tor` user can access it.
*   **`RUN echo "HiddenServiceDir ... " >> /etc/tor/torrc`**: Configures Tor to expose the Onion Service. The critical line is `HiddenServicePort 80 127.0.0.1:8080`, which maps external port 80 (on the Tor network) to the internal Nginx port.  Also, sets Tor to log to a file.
*   **`RUN mkdir -p /var/www/dummy ...`**: Creates a dummy website that listens on port 8081.  This acts as a fallback server if Tor or Nginx fails.
*   **`VOLUME /var/lib/tor/onion_service`**: Declares a volume for the Onion Service directory. This is crucial if you want to persist the Onion Service key across container restarts or redeployments.  **Important:**  If the platform you are using does not persist volumes, then you will lose the private key and the onion address will change every time your service is restarted.
*   **`CMD service nginx start && service tor start ...`**: Starts Nginx and Tor when the container starts.  It also prints the Onion address to the console and runs the dummy server in the background.  The `tail -f /var/log/tor/log` command keeps the container running by tailing the Tor log.

#### **3.3 Deploying on Koyeb**

Koyeb is a serverless platform that makes it easy to deploy Docker containers.

**Steps:**

1.  **Create a Koyeb Account**: Sign up for a Koyeb account.
2.  **Create a New App**:  In the Koyeb dashboard, create a new app.
3.  **Choose a Deployment Method**: Select "Docker Image" as the deployment method.
4.  **Specify the Docker Image**: Provide the Docker image name you built from the Dockerfile.
5.  **Configure Port(s)**:  Koyeb automatically detects the exposed ports. If not, specify port `8081` for the dummy server.  You **cannot** directly expose the Tor port. Tor handles external connections.
6.  **Deploy the App**: Deploy the app.

#### **3.4 Deploying on Render**

Render is another platform that simplifies deploying Docker containers.

**Steps:**

1.  **Create a Render Account**: Sign up for a Render account.
2.  **Create a New Web Service**:  In the Render dashboard, create a new web service.
3.  **Connect to a Git Repository**: Connect your Render service to the Git repository containing your Dockerfile and static website files. Render will automatically build and deploy the image.
4.  **Configure the Service**:
    *   Specify the Dockerfile path.
    *   Choose a name for your service.
    *   Select a region.
    *   Choose an instance type.
5.  **Deploy the Service**: Deploy the service.  Render will automatically build the Docker image and deploy it.

#### **3.5 General Deployment Notes**

*   **Monitoring Tor Logs**:  Regardless of the platform you choose, it's crucial to monitor the Tor logs for errors and warnings.  These logs will provide valuable insights into the health of your Onion Service.
*   **Security**: Hosting an Onion Service requires careful consideration of security. Keep Tor and other software up-to-date, and follow best practices for server security. Consider disabling SSH access to your container to minimize the attack surface.
*   **Vanity Addresses**: If you are using `mkp224o` to generate a vanity address, understand that this process can take a significant amount of time and resources.  For faster deployments, consider using a randomly generated Onion address.  If you skip the vanity generation, then instead of  `/mkp224o/mkp224o -d /var/lib/tor/onion_service -T 3 -n 1 yourpatternhere`, run `tor --keygen HiddenServiceDir /var/lib/tor/onion_service/`.

### **4. Security Best Practices for Onion Services**

Securing an Onion Service involves more than just setting up Tor. You need to protect the underlying website and the server it runs on.

#### **4.1 Web Application Security**

*   **HTTPS within Tor is Still Important**:  Even though traffic within the Tor network is encrypted, using HTTPS on your website provides an additional layer of security and integrity.  Configure Nginx to serve content over HTTPS if possible, using self-signed certificates or Let's Encrypt.  However, note that Let's Encrypt might require a domain name, which defeats the purpose of an onion service.
*   **Input Validation and Output Encoding**:  Protect against common web vulnerabilities like Cross-Site Scripting (XSS) and SQL Injection by carefully validating user input and encoding output.
*   **Regular Security Audits**:  Conduct regular security audits of your website's code and configuration to identify and address potential vulnerabilities.
*   **Disable Unnecessary Features**:  Disable any unnecessary features or services on your web server to reduce the attack surface.
*   **Content Security Policy (CSP)**: Implement a strong Content Security Policy (CSP) to prevent XSS attacks.

#### **4.2 Server Security**

*   **Keep Software Up-to-Date**:  Regularly update Tor, Nginx, and other software packages to patch security vulnerabilities.
*   **Strong Passwords and SSH Keys**:  Use strong passwords or SSH keys for accessing the server. Consider disabling password-based authentication altogether and only allowing SSH key-based authentication.
*   **Firewall Configuration**:  Configure a firewall to restrict access to the server. Only allow necessary ports to be open. Since you are using the Tor network, expose ports 80 and 443 are not required.
*   **Limit User Privileges**:  Run web server processes with the least amount of privilege necessary to perform their functions.
*   **Regular Security Scans**:  Perform regular security scans to identify and address potential vulnerabilities in the server's configuration.
*   **Monitor System Logs**:  Regularly monitor system logs for suspicious activity.
*   **Disable SSH**: Consider disabling SSH access. This can be done at the dockerfile level or within the platform's settings.

#### **4.3 Tor Security**

*   **Keep Tor Updated**:  Ensure that you are running the latest version of Tor to benefit from security updates.
*   **Monitor Tor Logs**:  Regularly monitor the Tor logs for errors and warnings.
*   **HiddenServiceAuthorizeClient**: This is a more advanced feature, but if you are only providing service to a limited number of clients, you can require client authorization to access the onion service.
*   **HiddenServiceVersion**:  Ensure that you are only using `HiddenServiceVersion 3` in your `torrc` file.
*   **Understand Tor's Limitations**: Tor provides anonymity, but it does not guarantee security. Be aware of the limitations of Tor and take appropriate security measures.

### **5. Conclusion**

Hosting Onion v3 websites on SaaS platforms like Koyeb and Render offers a convenient and scalable way to provide anonymous services. By leveraging Docker containers and understanding the nuances of each platform, you can deploy and manage your Onion Service effectively. However, it's crucial to prioritize security and follow best practices for web application, server, and Tor security to protect your service and its users.  Remember to carefully consider persistence and logging when choosing a platform. The key to a successful deployment lies in a well-configured Dockerfile and a solid understanding of the underlying platform's capabilities.
