import { setupServer } from "msw/node";

export const API_URL = "http://api.test";

export const server = setupServer();
