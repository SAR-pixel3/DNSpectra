
# Build DNSpectra using Maven and Java 26
FROM maven:3.9-eclipse-temurin-26 AS build

WORKDIR /app

# Copy Maven project files
COPY pom.xml .
COPY src ./src

# Build the Spring Boot application
RUN mvn -B -DskipTests package


# Run DNSpectra using Java 26
FROM eclipse-temurin:26-jre

WORKDIR /app

# Copy the built application
COPY --from=build /app/target/*.jar app.jar

# Copy the trained machine-learning models
COPY models ./models

# Render provides the port through the PORT environment variable
EXPOSE 10000

CMD ["java", "-jar", "app.jar"]
