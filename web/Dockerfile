FROM node:20-alpine

WORKDIR /app

# Copy package files
COPY package.json package-lock.json* ./

# Install dependencies
RUN npm ci

# Copy the rest of the application
COPY . .

# Expose the port
EXPOSE 3000

# Start the Next.js development server
# Note: For production, this should be built first, but we use dev for now
CMD ["npm", "run", "dev"]
