import { Router, type IRouter } from "express";
import healthRouter from "./health.js";
import drugRouter from "./drug.js";

const router: IRouter = Router();

router.use(healthRouter);
router.use(drugRouter);

export default router;
