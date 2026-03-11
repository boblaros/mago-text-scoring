FROM node:20-alpine AS base

WORKDIR /srv/app/frontend

COPY frontend/package.json ./
RUN npm install

COPY frontend /srv/app/frontend

FROM base AS dev

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]

FROM base AS build

ARG VITE_API_BASE_URL=http://localhost:8000/api/v1
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}

RUN npm run build

FROM nginx:1.27-alpine AS prod

COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /srv/app/frontend/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]

